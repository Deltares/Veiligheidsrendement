def add_computation_scenario_id(
    source: list[dict], computation_scenario_id: int
) -> None:
    """
    Adds the computation scenario id to the data entries

    Args:
        source (list[dict]): The collection of data entries to add the computation scenario id to.
        computation_scenario_id (int): The value of the computation scenario id to set.
    """
    for item in source:
        item["computation_scenario_id"] = computation_scenario_id
