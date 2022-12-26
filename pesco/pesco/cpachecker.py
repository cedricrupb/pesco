import os

from pathlib import Path
CURRENT_PATH = Path(__file__).parent.absolute()

# Helper -------------------------------------------------

def find_library(lib_path):
    current_path = str(CURRENT_PATH)
    while "pesco" in current_path:
        target_path = os.path.join(current_path, lib_path)
        if os.path.exists(target_path): return target_path
        current_path = os.path.dirname(current_path)

    raise ValueError("Cannot find lib %s in the current domain: %s" % (lib_path, CURRENT_PATH))

# CONSTANTS ---------------------------------------------

CPA_CONFIGS = {
    "symbolic": "pesco23-symex",
    "va"      : "pesco23-va",
    "vaitp"   : "pesco23-va-itp",
    "ki"      : "pesco23-ki",
    "pa"      : "pesco23-pa",
    "bmc"     : "pesco23-bmc",
    "bam"     : "pesco23-bam",
}

BASE_COMPONENTS = {
    "symbolic": "pesco23-symbolicExecution-full.properties",
    "va"      : "pesco23-valueAnalysis-full.properties",
    "vaitp"   : "pesco23-valueAnalysis-itp-full.properties",
    "ki"      : "pesco23-kInduction-full.properties",
    "pa"      : "pesco23-predicateAnalysis-full.properties",
    "bmc"     : "pesco23-bmc-full.properties",
    "bam"     : "pesco23-bam-full.properties",
}

CPACHECKER = find_library(os.path.join("lib", "cpachecker"))

# Functions --------------------------------------------------

def _parse_composition(composition):
    if isinstance(composition, list): return composition

    if "," in composition:
        composition = composition.split(",")
    else:
        composition = [composition]
    
    def _interpret_single(config):
        config = config.strip()
        if ":" not in config: config = config+":900"
        config, time = config.split(":")
        time = int(time)
        return (config, time)

    return list(map(_interpret_single, composition))


def _generate_component(config, time):
    if time >= 800: return BASE_COMPONENTS[config]

    base_component_rel = BASE_COMPONENTS[config]
    base_component_path = os.path.join(CPACHECKER, "config", "components", base_component_rel)
    
    new_component_rel, _ = os.path.splitext(base_component_rel)
    new_component_rel    = f"{new_component_rel}-{time}.properties"
    new_component_path   = os.path.join(CPACHECKER, "config", "components", new_component_rel)

    if os.path.exists(new_component_path): return new_component_rel

    with open(base_component_path, "r") as i:
        content = i.read()

    content += f"""
    limits.time.cpu = {time}s
    limits.time.cpu::required = {time}s
    """

    with open(new_component_path, "w") as o:
        o.write(content)

    return new_component_rel


def _generate_strategy_name(composition):
    strategy_components = ["pesco23"]

    for config, time in composition:
        strategy_components.append(f"{config}{time}")

    return "-".join(strategy_components)


def generate_cpa_config(composition):
    if composition in CPA_CONFIGS:
        return CPA_CONFIGS[composition]
    
    composition = _parse_composition(composition)

    if len(composition) == 1 and composition[0][1] >= 800:
        composition = composition[0][0]
        if composition in CPA_CONFIGS:
            return CPA_CONFIGS[composition]
    
    components = []
    for config, time in composition:
        components.append(_generate_component(config, time))

    analyses = ", \\\n".join(f"components/{component}" for component in components)

    strategy_name = _generate_strategy_name(composition)
    verifier_strategy_template = os.path.join(CPACHECKER, "config", "pesco23-strategy-template.properties")
    verifier_strategy_path = os.path.join(CPACHECKER, "config", f"{strategy_name}.properties")

    if os.path.exists(verifier_strategy_path): return strategy_name

    with open(verifier_strategy_template, "r") as i:
        template = i.read()

    verifier_strategy = template.replace("__ANALYSES__", analyses)

    with open(verifier_strategy_path, "w") as o:
        o.write(verifier_strategy)

    return strategy_name


