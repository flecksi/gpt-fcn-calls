from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2

def optimize_model(inputs:dict, outputs:dict, verbose:bool=True) -> dict:
    model = cp_model.CpModel()

    input_vars = {}

    input_max_abs_limit = {}

    for i,input_dict in inputs.items():
        input_max_abs_limit[i] = max([abs(input_dict["min"]),abs(input_dict["max"])])
        input_vars[i] = model.new_int_var(input_dict["min"], input_dict["max"], i)
        if input_dict["locked"]:
            model.add(input_vars[i]==input_dict["val"])
        else:
            model.add_hint(input_vars[i],input_dict["val"])

    cost = 0

    output_vars = {}
    output_error_vars = {}

    for o,output_dict in outputs.items():
        if o=="x":
            o_lim = input_max_abs_limit["a"] + input_max_abs_limit["b"] + input_max_abs_limit["c"]
        elif o=="y":
            o_lim = input_max_abs_limit["a"] * input_max_abs_limit["b"]
        elif o=="z":
            o_lim = input_max_abs_limit["b"] + input_max_abs_limit["c"]
        else:
            raise ValueError(f"unknown output variable '{o}', must be x,y or z")
                
        o_var =  model.new_int_var(-o_lim,o_lim,o)
        output_vars[o] = o_var
        o_var_error = model.new_int_var(-2*o_lim,2*o_lim,f"{o}_err")
        output_error_vars[o] = o_var_error

        if o=="x":
            model.add(o_var==input_vars["a"] + input_vars["b"] + input_vars["c"])
        elif o=="y":
            model.add_multiplication_equality(o_var,[input_vars["a"], input_vars["b"]])
        elif o=="z":
            model.add(o_var==input_vars["b"] - input_vars["c"])
        else:
            raise ValueError(f"unknown output variable '{o}', must be x,y or z")
        model.add_abs_equality(o_var_error,o_var-output_dict["target"])
        if output_dict["active"]:
            cost += o_var_error

    model.minimize(cost)
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model_pb2.CpSolverStatus.OPTIMAL:
        status_string = "Optimal"
    elif status == cp_model_pb2.CpSolverStatus.FEASIBLE:
        status_string = "Feasible"
    elif status == cp_model_pb2.CpSolverStatus.INFEASIBLE:
        status_string = "Infeasible"
    elif status == cp_model_pb2.CpSolverStatus.MODEL_INVALID:
        status_string = "Model Invalid"
    elif status == cp_model_pb2.CpSolverStatus.UNKNOWN:
        status_string = "Unknown"
    else:
        raise RuntimeError(f"CP_Sat solver status not known: {status}")

    if verbose:
        print("Solution:")
        print(f"  Status={status_string}")
        print(f"  Total cost={solver.objective_value}")

    rv_dict = {
        "inputs":{},
        "outputs":{},
        "cost":solver.objective_value,
        "status":status_string
    }

    if verbose: print("\nInputs:")
    for input_name,input_dict in inputs.items():
        i_val = solver.Value(input_vars[input_name])
        rv_dict["inputs"][input_name] = i_val
        if verbose: print(f"  Input {input_name}={i_val} (initial={input_dict["val"]}, locked={input_dict["locked"]}, min={input_dict["min"]}, max={input_dict["max"]})")
    
    if verbose: print("\nOutputs:")
    for output_name,output_dict in outputs.items():
        o_val=solver.Value(output_vars[output_name])
        o_target=output_dict["target"]
        rv_dict["outputs"][output_name] = o_val
        
        if verbose: print(f"  Output {output_name}={o_val} (target={o_target}, active={output_dict["active"]}, delta={o_val-o_target}, part cost={abs(o_val-o_target) if output_dict["active"] else '-'})")

    return rv_dict

def calc_model_output(inputs:dict) -> dict:
    return dict(
        x=inputs["a"]+inputs["b"]+inputs["c"],
        y=inputs["a"]*inputs["b"],
        z=inputs["b"]-inputs["c"],
    )

if __name__ == "__main__":
    inputs = dict(
        a=dict(min=-10,max=10,val=3,locked=False),
        b=dict(min=-10,max=10,val=6,locked=False),
        c=dict(min=-10,max=11,val=3,locked=True),
    )

    outputs = dict(
        x=dict(target=-15, active=False),
        y=dict(target=65, active=True),
        z=dict(target=-5, active=True),
    )

    opti_result_dict = optimize_model(inputs=inputs, outputs=outputs, verbose=True)
    print(f"\n{calc_model_output(inputs=opti_result_dict['inputs'])=}")