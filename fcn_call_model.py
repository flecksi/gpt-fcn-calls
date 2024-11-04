from pydantic import BaseModel
import openai
import pandas as pd

from model_solver import optimize_model, calc_model_output


class ModelInput(BaseModel):
    name: str
    descr: str
    unit: str
    value: int
    min: int
    max: int
    locked: bool


class ModelOutput(BaseModel):
    name: str
    descr: str
    unit: str
    value: int
    target: int
    target_active: bool

    @property
    def cost(self) -> int:
        return abs(self.value - self.target) if self.target_active else 0


class Model(BaseModel):
    name: str
    descr: str
    inputs: list[ModelInput]
    outputs: list[ModelOutput]

    @property
    def df_inputs(self) -> pd.DataFrame:
        records = [i.model_dump() for i in self.inputs]
        return pd.DataFrame.from_records(records)

    @property
    def df_outputs(self) -> pd.DataFrame:
        records = [o.model_dump() for o in self.outputs]
        df = pd.DataFrame.from_records(records)
        df["part_cost"] = [o.cost for o in self.outputs]
        return df

    def update_outputs(self) -> None:
        output_dict = calc_model_output({i.name: i.value for i in self.inputs})
        for o in self.outputs:
            o.value = output_dict[o.name]

    def optimize(self, verbose: bool = False) -> str:
        inputs = {
            i.name: {"min": i.min, "max": i.max, "val": i.value, "locked": i.locked}
            for i in self.inputs
        }

        outputs = {
            o.name: {"target": o.target, "active": o.target_active}
            for o in self.outputs
        }

        result_dict = optimize_model(inputs=inputs, outputs=outputs, verbose=verbose)
        for i_opt_name, i_opt_val in result_dict["inputs"].items():
            for i in self.inputs:
                if i.name == i_opt_name:
                    i.value = i_opt_val

        self.update_outputs()
        return result_dict["status"]

    @property
    def cost(self) -> int:
        return sum([o.cost for o in self.outputs])


model_inputs = [
    ModelInput(
        name="a",
        descr="first model input",
        unit="kg",
        value=1,
        min=0,
        max=10,
        locked=False,
    ),
    ModelInput(
        name="b",
        descr="second model input",
        unit="m",
        value=9,
        min=-2,
        max=19,
        locked=True,
    ),
    ModelInput(
        name="c",
        descr="third model input",
        unit="kg",
        value=3,
        min=0,
        max=11,
        locked=True,
    ),
]

model_outputs = [
    ModelOutput(
        name="x",
        descr="first model output",
        unit="s",
        target=0,
        target_active=False,
        value=0,
    ),
    ModelOutput(
        name="y",
        descr="second model output",
        unit="N",
        target=19,
        target_active=True,
        value=0,
    ),
    ModelOutput(
        name="z",
        descr="third model output",
        unit="Nm",
        target=0,
        target_active=True,
        value=0,
    ),
]

model = Model(
    name="M1",
    descr="A test-model for development and testing",
    inputs=model_inputs,
    outputs=model_outputs,
)


class F_optimize_model(BaseModel):
    timeout_seconds: int


class F_plot_sensitivity(BaseModel):
    inputs: list[str]
    outputs: list[str]


tools = [
    openai.pydantic_function_tool(
        Model, name="update_model", description="function to update a model."
    ),
    openai.pydantic_function_tool(
        F_optimize_model,
        name="optimize_model",
        description="function to optimize a model (i.e. modify input values so that outputs reach their targets, if active). default timeout=-1",
    ),
    openai.pydantic_function_tool(
        F_plot_sensitivity,
        name="plot_sensitivity",
        description="function to create a sensitivity plot. inputs must be a list of model-input names. outputs must be a list of model-output names. lists may not be empty. if not specified use all inputs and outputs",
    ),
]


def sysprompt(model: Model) -> str:
    return f"You are an assistant that helps the user interacting with a model. If the user wants to change anything in the model, use the given tool 'update_model' and try to only invoke it a single time. The model is defined as follows:\n{model.model_dump_json(indent=2)}"
