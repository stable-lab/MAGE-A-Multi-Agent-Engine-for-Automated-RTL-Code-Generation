import os
import re
import sys
import traceback
from typing import List, Tuple

from llama_index.core.llms import LLM

from .log_utils import get_logger, set_log_dir, switch_log_to_file, switch_log_to_stdout
from .rtl_editor import RTLEditor
from .rtl_generator import RTLGenerator
from .sim_judge import SimJudge
from .sim_reviewer import SimReviewer
from .tb_generator import TBGenerator
from .token_counter import TokenCounter, TokenCounterCached

logger = get_logger(__name__)


class TopAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.token_counter = (
            TokenCounterCached(llm)
            if TokenCounterCached.is_cache_enabled(llm)
            else TokenCounter(llm)
        )
        self.sim_max_retry = 4
        self.rtl_max_candidates = 20
        self.rtl_selected_candidates = 2
        self.is_ablation = False
        self.redirect_log = False
        self.output_path = "./output"
        self.log_path = "./log"
        self.golden_tb_path: str | None = None
        self.golden_rtl_blackbox_path: str | None = None
        self.tb_gen: TBGenerator | None = None
        self.rtl_gen: RTLGenerator | None = None
        self.sim_reviewer: SimReviewer | None = None
        self.sim_judge: SimJudge | None = None
        self.rtl_edit: RTLEditor | None = None

    def set_output_path(self, output_path: str) -> None:
        self.output_path = output_path

    def set_log_path(self, log_path: str) -> None:
        self.log_path = log_path

    def set_ablation(self, is_ablation: bool) -> None:
        self.is_ablation = is_ablation

    def set_redirect_log(self, new_value: bool) -> None:
        self.redirect_log = new_value
        if self.redirect_log:
            switch_log_to_file()
        else:
            switch_log_to_stdout()

    def write_output(self, content: str, file_name: str) -> None:
        assert self.output_dir_per_run
        with open(f"{self.output_dir_per_run}/{file_name}", "w") as f:
            f.write(content)

    def run_instance(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.tb_gen
        assert self.rtl_gen
        assert self.sim_reviewer
        assert self.sim_judge
        assert self.rtl_edit

        self.tb_gen.reset()
        self.tb_gen.set_golden_tb_path(self.golden_tb_path)
        if not self.golden_tb_path:
            logger.info("No golden testbench provided")
        testbench, interface = self.tb_gen.chat(spec)
        logger.info("Initial tb:")
        logger.info(testbench)
        logger.info("Initial if:")
        logger.info(interface)
        self.write_output(testbench, "tb.sv")
        self.write_output(interface, "if.sv")
        self.rtl_gen.reset()
        logger.info(spec)

        is_syntax_pass, rtl_code = self.rtl_gen.chat(
            input_spec=spec,
            testbench=testbench,
            interface=interface,
            rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
        )
        if not is_syntax_pass:
            return False, rtl_code
        self.write_output(rtl_code, "rtl.sv")
        logger.info("Initial rtl:")
        logger.info(rtl_code)

        tb_need_fix = True
        rtl_need_fix = True
        sim_log = ""
        for i in range(self.sim_max_retry):
            # run simulation judge, overwrite is_sim_pass
            is_sim_pass, sim_mismatch_cnt, sim_log = self.sim_reviewer.review()
            if is_sim_pass:
                tb_need_fix = False
                rtl_need_fix = False
                break
            self.sim_judge.reset()
            tb_need_fix = self.sim_judge.chat(spec, sim_log, rtl_code, testbench)
            if tb_need_fix:
                self.tb_gen.reset()
                if i == 0:
                    self.tb_gen.gen_display_queue = False
                    logger.info("Fallback from display queue to display moment")
                else:
                    self.tb_gen.set_failed_trial(sim_log, rtl_code, testbench)

                testbench, _ = self.tb_gen.chat(spec)
                self.write_output(testbench, "tb.sv")
                logger.info("Revised tb:")
                logger.info(testbench)
            else:
                break

        assert not tb_need_fix, f"tb_need_fix should be False. sim_log: {sim_log}"

        candidates_info: List[Tuple[str, int, str]] = []
        if rtl_need_fix:
            # Candidates Generation
            assert (
                sim_mismatch_cnt > 0
            ), f"rtl_need_fix should be True only when sim_mismatch_cnt > 0. sim_log: {sim_log}"
            self.rtl_gen.reset()
            candidates = [
                self.rtl_gen.chat(
                    input_spec=spec,
                    testbench=testbench,
                    interface=interface,
                    rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
                    enable_cache=True,
                )
            ]  # Write Cache
            if self.rtl_max_candidates > 1:
                candidates += self.rtl_gen.gen_candidates(
                    input_spec=spec,
                    testbench=testbench,
                    interface=interface,
                    rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv"),
                    candidates_num=self.rtl_max_candidates - 1,
                    enable_cache=True,
                )
            for i in range(self.rtl_max_candidates):
                logger.info(
                    f"Candidate generation: round {i + 1} / {self.rtl_max_candidates}"
                )
                is_syntax_pass_candiate, rtl_code_candidate = candidates[i]
                if not is_syntax_pass_candiate:
                    continue
                self.write_output(rtl_code_candidate, "rtl.sv")
                is_sim_pass_candidate, sim_mismatch_cnt_candidate, sim_log_candidate = (
                    self.sim_reviewer.review()
                )
                if is_sim_pass_candidate:
                    rtl_code = rtl_code_candidate
                    sim_mismatch_cnt = sim_mismatch_cnt_candidate
                    sim_log = sim_log_candidate
                    rtl_need_fix = False
                    break
                candidates_info.append(
                    (rtl_code_candidate, sim_mismatch_cnt_candidate, sim_log_candidate)
                )

        candidates_info.sort(key=lambda x: x[1])
        candidates_info_unique_sign = set()
        candidates_info_unique = []
        for candidate in candidates_info:
            if candidate[1] not in candidates_info_unique_sign:
                candidates_info_unique_sign.add(candidate[1])
                candidates_info_unique.append(candidate)

        if rtl_need_fix:
            # Editor iteration
            for i in range(self.rtl_selected_candidates):
                logger.info(
                    f"Selected candidate: round {i + 1} / {self.rtl_selected_candidates}"
                )
                i = i % len(candidates_info_unique)
                rtl_code, sim_mismatch_cnt, sim_log = candidates_info_unique[i]
                with open(f"{self.output_dir_per_run}/rtl.sv", "w") as f:
                    f.write(rtl_code)
                self.rtl_edit.reset()
                is_sim_pass, rtl_code = self.rtl_edit.chat(
                    spec=spec,
                    output_dir_per_run=self.output_dir_per_run,
                    sim_failed_log=sim_log,
                    sim_mismatch_cnt=sim_mismatch_cnt,
                )
                if is_sim_pass:
                    rtl_need_fix = False
                    break

        if not is_sim_pass:  # Run if keep failing before last try
            is_sim_pass, _, _ = self.sim_reviewer.review()

        return is_sim_pass, rtl_code

    def run_instance_ablation(self, spec: str) -> Tuple[bool, str]:
        """
        Run a single instance of the benchmark in ablation mode
        Return value:
        - is_pass: bool, whether the instance passes the golden testbench
        - rtl_code: str, the generated RTL code
        """
        assert self.rtl_gen

        self.rtl_gen.reset()
        logger.info(spec)
        # Current ablation: only run RTL generation with syntax check
        is_syntax_pass, rtl_code = self.rtl_gen.ablation_chat(
            input_spec=spec, rtl_path=os.path.join(self.output_dir_per_run, "rtl.sv")
        )
        self.write_output(rtl_code, "rtl.sv")
        return is_syntax_pass, rtl_code

    def _run(self, spec: str) -> Tuple[bool, str]:
        try:
            if os.path.exists(f"{self.output_dir_per_run}/properly_finished.tag"):
                os.remove(f"{self.output_dir_per_run}/properly_finished.tag")
            self.token_counter.reset()
            self.sim_reviewer = SimReviewer(
                self.output_dir_per_run,
                self.golden_rtl_blackbox_path,
            )
            self.rtl_gen = RTLGenerator(self.token_counter)
            self.tb_gen = TBGenerator(self.token_counter)
            self.sim_judge = SimJudge(self.token_counter)
            self.rtl_edit = RTLEditor(
                self.token_counter, sim_reviewer=self.sim_reviewer
            )
            ret = (
                self.run_instance(spec)
                if not self.is_ablation
                else self.run_instance_ablation(spec)
            )
            self.token_counter.log_token_stats()
            with open(f"{self.output_dir_per_run}/properly_finished.tag", "w") as f:
                f.write("1")
        except Exception:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            ret = False, f"Exception: {exc_info[1]}"
        return ret

    def run(
        self,
        benchmark_type_name: str,
        task_id: str,
        spec: str,
        golden_tb_path: str | None = None,
        golden_rtl_blackbox_path: str | None = None,
    ) -> Tuple[bool, str]:
        self.golden_tb_path = golden_tb_path
        self.golden_rtl_blackbox_path = golden_rtl_blackbox_path
        log_dir_per_run = f"{self.log_path}/{benchmark_type_name}_{task_id}"
        self.output_dir_per_run = f"{self.output_path}/{benchmark_type_name}_{task_id}"
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.output_dir_per_run, exist_ok=True)
        set_log_dir(log_dir_per_run)
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "w") as f:
                sys.stdout = f
                sys.stderr = f
                result = self._run(spec)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        else:
            result = self._run(spec)
        # Redirect log contains format with rich text.
        # Provide a rich-free version for log parsing or less viewing.
        if self.redirect_log:
            with open(f"{log_dir_per_run}/mage_rtl.log", "r") as f:
                content = f.read()
            content = re.sub(r"\[.*?m", "", content)
            with open(f"{log_dir_per_run}/mage_rtl_rich_free.log", "w") as f:
                f.write(content)
        return result
