import json

from .utils import _path_to_task_id


def load_evaluator_from_file(file_path):

    if file_path.endswith("json"):
        with open(file_path, "r") as f:
            result_cache = json.load(f)
    else:
        result_cache = {}

        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                
                for file_entry in entry:
                    task_id = _path_to_task_id(file_entry["task_file"])
                    if task_id not in result_cache: result_cache[task_id] = {}
                    result_entry = result_cache[task_id]

                    result_entry[file_entry["tool"]] = {
                        "status": file_entry["status"],
                        "verdict": file_entry["verdict"],
                        "cputime": file_entry["cputime"],
                        "memory" : file_entry["memory"],
                    }

    return Evaluator(result_cache)




class Evaluator:

    def __init__(self, results):
        self._result_cache = results
        self._tools = set.union(*[set(v.keys()) for v in results.values()])

    def tools(self):
        return self._tools
    
    def instances(self):
        return list(self._result_cache.keys())

    def union(self, other):
        return union(self, other)

    def _prepare_portfolio(self, portfolio_desc, timelimit):

        def _parse(desc):
            if isinstance(desc, str):
                desc = _parse_portfolio_desc(desc, timelimit=timelimit)
            return desc

        if not isinstance(portfolio_desc, dict):
            desc = _parse(portfolio_desc)
            portfolio_desc = {k: desc for k in self.instances()}
        else:
            portfolio_desc = {k: _parse(v) for k, v in portfolio_desc.items()}

        return portfolio_desc


    def _should_stop(self, exec_result):
        if exec_result == "unknown"        : return False
        if "error" in exec_result.lower()  : return False
        if exec_result == "EXCEPTION"      : return False
        if exec_result == "done"           : return False

        return True


    def eval(self, portfolio_desc, timelimit = 900):
        portfolio_desc = self._prepare_portfolio(portfolio_desc, timelimit)

        exec_results = {}

        for instance, portfolio in portfolio_desc.items():
            real_results = self._result_cache[instance]
            verdict = set(str(r["verdict"]).lower() for r in real_results.values())
            assert len(verdict) == 1, "A task cannot have more than on verdict"
            verdict = next(iter(verdict))

            remain_time = timelimit

            for tool, runtime in portfolio:
                exec_result = "unknown"
                tool_result = real_results.get(tool, {"status": "unknown", "cputime": 1e+9})

                available_time = min(remain_time, runtime)

                # Decide whether tool computes result
                if tool_result["cputime"] < available_time:
                    exec_result = tool_result["status"]
                    runtime = tool_result["cputime"]
                
                remain_time -= min(runtime, available_time)
                if self._should_stop(exec_result) or remain_time <= 0: break

            if remain_time == 0: exec_result = "TIMEOUT"
            runtime = timelimit - remain_time
            exec_results[instance] = (exec_result, verdict, runtime)
        
        return EvaluationResult(exec_results)


class EvaluationResult:

    def __init__(self, results):
        self.results = results

    def score(self):
        return sum(1 if r[0].startswith(r[1]) else 0 for r in self.results.values())

    def __getitem__(self, instance):
        return self.results[instance]

    def __iter__(self):
        return iter((r[0], r[1][0], r[1][1], r[1][2]) for r in self.results.items())

    def __repr__(self):
        return f"Result( {self.score()} / {len(self.results)} )"


# API methods --------------------------------

def union(eval1, eval2):
    result1, result2 = eval1._result_cache, eval2._result_cache

    common_keys = set.union(set(result1.keys()), set(result2.keys()))
    new_results = {}

    for key in common_keys:
        r1 = dict(result1.get(key, {}))
        r2 = dict(result2.get(key, {}))
        r1.update(r2)

        new_results[key] = r1

    return Evaluator(new_results)
    


# Helper --------------------------------

def _parse_portfolio_desc(portfolio_desc, timelimit = 900):
    if "," in portfolio_desc:
        portfolio_desc = [r.strip() for r in portfolio_desc.split(",")]
    else:
        portfolio_desc = [portfolio_desc]

    def _parse_time(d):
        if ":" in d:
            name, time = d.split(":")
            time  = min(int(time), timelimit)
        else:
            name, time = d, timelimit
        
        return (name, time)

    return list(map(_parse_time, portfolio_desc))
    

    
