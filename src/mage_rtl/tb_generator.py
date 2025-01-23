import json
from typing import Dict, List, Tuple

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from .log_utils import get_logger
from .prompts import FAILED_TRIAL_PROMPT, ORDER_PROMPT, TB_4_SHOT_EXAMPLES
from .token_counter import TokenCounter, TokenCounterCached
from .utils import add_lineno

logger = get_logger(__name__)

SYSTEM_PROMPT = r"""
You are an expert in SystemVerilog design.
You can always write SystemVerilog code with no syntax errors and always reach correct functionality.
"""

NON_GOLDEN_TB_PROMPT = r"""
In order to test a module generated with the given natural language specification:
1. Please write an IO interface for that module;
2. Please write a testbench to test the module.

The module interface should EXACTLY MATCH the description in input_spec.
(Including the module name, input/output ports names, and their types)

<input_spec>
{input_spec}
</input_spec>

The testbench should:
1. Instantiate the module according to the IO interface;
2. Generate input stimulate signals and expected output signals according to input_spec;
3. Apply the input signals to the module, count the number of mismatches between the output signals with the expected output signals;
4. Every time when a check occurs, no matter match or mismatch, display input signals, output signals and expected output signals;
5. When simulation ends, ADD DISPLAY "SIMULATION PASSED" if no mismatch occurs, otherwise display:
    "SIMULATION FAILED - x MISMATCHES DETECTED, FIRST AT TIME y".
6. To avoid ambiguity, please use the reverse edge to do output check. (If RTL runs at posedge, use negedge to check the output)
7. For pure combinational module (especially those without clk),
    the expected output should be checked at the exact moment when the input is changed;
8. Avoid using keyword "continue"

Try to understand the requirements above and give reasoning steps in natural language to achieve it.
In addition, try to give advice to avoid syntax error.
An SystemVerilog module always starts with a line starting with the keyword 'module' followed by the module name.
It ends with the keyword 'endmodule'.

{examples_prompt}

Please also follow the display prompt below:
{display_prompt}
"""

GOLDEN_TB_PROMPT = r"""
In order to test a module generated with the given natural language specification:
1. Please write an IO interface for that module;
2. Please improve the given golden testbench to test the module.

The module interface should EXACTLY MATCH the description in input_spec.
(Including the module name, input/output ports names, and their types)

<input_spec>
{input_spec}
</input_spec>

To improve the golden testbench, you should add more display to it, while keeping the original functionality.
In detail, the testbench you generated should:
1. MAINTAIN the EXACT SAME functionality, interface and module instantiation  as the golden testbench;
2. If the golden testbench contradicts the input_spec, ALWAYS FOLLOW the golden testbench;
3. MAINTAIN the original logic of error counting;
4. When simulation ends, ADD DISPLAY "SIMULATION PASSED" if no mismatch occurs, otherwise display:
    "SIMULATION FAILED - x MISMATCHES DETECTED, FIRST AT TIME y".
Please also follow the display prompt below:
{display_prompt}


Try to understand the requirements above and give reasoning steps in natural language to achieve it.
In addition, try to give advice to avoid syntax error.
An SystemVerilog module always starts with a line starting with the keyword 'module' followed by the module name.
It ends with the keyword 'endmodule'.

Below is the golden testbench code for the module generated with the given natural language specification.
<golden_testbench>
{golden_testbench}
<golden_testbench>
"""

DISPLAY_MOMENT_PROMPT = r"""
1. When the first mismatch occurs, display the input signals, output signals and expected output signals at that time.
2. For multiple-bit signals displayed in HEX format, also display the BINARY format if its width <= 64.
"""

DISPLAY_QUEUE_PROMPT = r"""
1. If module to test is sequential logic (like including an FSM):
    1.1. Store input signals, output signals, expected output signals and reset signals in a queue with MAX_QUEUE_SIZE;
        When the first mismatch occurs, display the queue content after storing it. Make sure the mismatched signal can be displayed.
    1.2. MAX_QUEUE_SIZE should be set according to the requirement of the module.
        For example, if the module has a 3-bit state, MAX_QUEUE_SIZE should be at least 2 ** 3 = 8.
        And if the module was to detect a pattern of 8 bits, MAX_QUEUE_SIZE should be at least (8 + 1) = 9.
        However, to control log size, NEVER set MAX_QUEUE_SIZE > 10.
    1.3. The clocking of queue and display should be same with the clocking of tb_match detection.
        For example, if 'always @(posedge clk, negedge clk)' is used to detect mismatch,
        It should also be used to push queue and display first error.
2. If module to test is combinational logic:
    When the first mismatch occurs, display the input signals, output signals and expected output signals at that time.
3. For multiple-bit signals displayed in HEX format, also display the BINARY format if its width <= 64.

<display_queue_example>
// Queue-based simulation mismatch display

reg [INPUT_WIDTH-1:0] input_queue [$];
reg [OUTPUT_WIDTH-1:0] got_output_queue [$];
reg [OUTPUT_WIDTH-1:0] golden_queue [$];
reg reset_queue [$];

localparam MAX_QUEUE_SIZE = 5;

always @(posedge clk, negedge clk) begin
    if (input_queue.size() >= MAX_QUEUE_SIZE - 1) begin
        input_queue.delete(0);
        got_output_queue.delete(0);
        golden_queue.delete(0);
        reset_queue.delete(0);
    end

    input_queue.push_back(input_data);
    got_output_queue.push_back(got_output);
    golden_queue.push_back(golden_output);
    reset_queue.push_back(rst);

    // Check for first mismatch
    if (got_output !== golden_output) begin
        $display("Mismatch detected at time %t", $time);
        $display("\nLast %d cycles of simulation:", input_queue.size());


        for (int i = 0; i < input_queue.size(); i++) begin
            if (got_output_queue[i] === golden_queue[i]) begin
                $display("Got Match at");
            end else begin
                $display("Got Mismatch at");
            end
            $display("Cycle %d, reset %b, input %h, got output %h, exp output %h",
                i,
                reset_queue[i],
                input_queue[i],
                got_output_queue[i],
                golden_queue[i]
            );
        end
    end

end
</display_queue_example>
"""


EXAMPLE_OUTPUT = {
    "reasoning": "All reasoning steps and advices to avoid syntax error",
    "interface": "The IO part of a SystemVerilog module, not containing the module implementation",
    "testbench": "The testbench code to test the module",
}


class TBOutputFormat(BaseModel):
    reasoning: str
    interface: str
    testbench: str


EXTRA_ORDER_GOLDEN_TB_PROMPT = r"""
Remember that if the golden testbench contradicts the input_spec, ALWAYS FOLLOW the golden testbench;
Especially if the input_spec say some input should not exist, but as long as the golden testbench uses it, you should use it.
Remember to display "SIMULATION PASSED" when simulation ends if no mismatch occurs, otherwise display "SIMULATION FAILED - x MISMATCHES DETECTED, FIRST AT TIME y".
Remember to add display for the FIRST mismatch, while maintaining the original logic of error counting;
ALWAYS generate the complete testbench, no matter how long it is.
Generate interface according to golden testbench, even if it contradicts the input_spec. Declare all ports as logic.
"""

EXTRA_ORDER_NON_GOLDEN_TB_PROMPT = r"""
For pattern detecter, if no specification is found in input_spec,
suppose the "detected" output will be asserted on the cycle AFTER the pattern appears in input.
Like when detecting pattern "11", should be like:
// Test case : Two consecutive ones
@(posedge clk); in_ = 1; expected_out = 0;
@(posedge clk); in_ = 1; expected_out = 0;
@(posedge clk); in_ = 0; expected_out = 1;
"""


class TBGenerator:
    def __init__(
        self,
        token_counter: TokenCounter,
    ):
        self.token_counter = token_counter
        self.failed_trial: List[ChatMessage] = []
        self.history: List[ChatMessage] = []
        self.golden_tb_path: str | None = None
        self.json_decode_max_trial = 3
        self.gen_display_queue = True

    def reset(self):
        self.history = []

    def set_golden_tb_path(self, golden_tb_path: str | None) -> None:
        self.golden_tb_path = golden_tb_path

    def set_failed_trial(
        self, failed_sim_log: str, previous_code: str, previous_tb: str
    ) -> None:
        cur_failed_trial = FAILED_TRIAL_PROMPT.format(
            failed_sim_log=failed_sim_log,
            previous_code=add_lineno(previous_code),
            previous_tb=add_lineno(previous_tb),
        )
        self.failed_trial.append(
            ChatMessage(content=cur_failed_trial, role=MessageRole.USER)
        )

    def generate(self, messages: List[ChatMessage]) -> ChatResponse:
        logger.info(f"TB generator input message: {messages}")
        resp, token_cnt = self.token_counter.count_chat(messages)
        logger.info(f"Token count: {token_cnt}")
        logger.info(f"{resp.message.content}")
        return resp

    def get_init_prompt_messages(self, input_spec: str) -> List[ChatMessage]:
        display_prompt = (
            DISPLAY_QUEUE_PROMPT if self.gen_display_queue else DISPLAY_MOMENT_PROMPT
        )
        if self.golden_tb_path:
            with open(self.golden_tb_path, "r") as f:
                golden_testbench = f.read()
            generation_content = GOLDEN_TB_PROMPT.format(
                input_spec=input_spec,
                golden_testbench=golden_testbench,
                display_prompt=display_prompt,
            )
        else:
            generation_content = NON_GOLDEN_TB_PROMPT.format(
                input_spec=input_spec,
                examples_prompt=TB_4_SHOT_EXAMPLES,
                display_prompt=display_prompt,
            )
        ret = [
            ChatMessage(content=SYSTEM_PROMPT, role=MessageRole.SYSTEM),
            ChatMessage(content=generation_content, role=MessageRole.USER),
        ]
        if self.failed_trial:
            ret.extend(self.failed_trial)
        return ret

    def get_order_prompt_messages(self) -> List[ChatMessage]:
        if self.golden_tb_path:
            order_prompt_message = ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                )
                + EXTRA_ORDER_GOLDEN_TB_PROMPT,
                role=MessageRole.USER,
            )
        else:
            order_prompt_message = ChatMessage(
                content=ORDER_PROMPT.format(
                    output_format="".join(json.dumps(EXAMPLE_OUTPUT, indent=4))
                )
                + EXTRA_ORDER_NON_GOLDEN_TB_PROMPT,
                role=MessageRole.USER,
            )

        return [order_prompt_message]

    def parse_output(self, response: ChatResponse) -> TBOutputFormat:
        try:
            output_json_obj: Dict = json.loads(response.message.content, strict=False)
            ret = TBOutputFormat(
                reasoning=output_json_obj["reasoning"],
                interface=output_json_obj["interface"],
                testbench=output_json_obj["testbench"],
            )
        except json.decoder.JSONDecodeError as e:
            ret = TBOutputFormat(
                reasoning=f"Json Decode Error: {str(e)}",
                interface="",
                testbench="",
            )
        return ret

    def chat(self, input_spec: str) -> Tuple[str, str]:
        if isinstance(self.token_counter, TokenCounterCached):
            self.token_counter.set_enable_cache(False)
        self.history = []
        self.token_counter.set_cur_tag(self.__class__.__name__)
        self.history.extend(self.get_init_prompt_messages(input_spec))
        for _ in range(self.json_decode_max_trial):
            response = self.generate(self.history + self.get_order_prompt_messages())
            resp_obj = self.parse_output(response)
            if not resp_obj.reasoning.startswith("Json Decode Error"):
                break
            error_msg = ChatMessage(role=MessageRole.USER, content=resp_obj.reasoning)
            self.history.extend([response.message, error_msg])
        if resp_obj.reasoning.startswith("Json Decode Error"):
            raise ValueError(
                f"Json Decode Error when decoding: {response.message.content}"
            )
        return (resp_obj.testbench, resp_obj.interface)
