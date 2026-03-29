def get_query() -> str:
    """
    This function is responsible for getting the user's query.
    It can be easily replaced if we want to implement a different interface (e.g., a web UI or an API).

    Returns:
        str: The user's query as a string.
    """
    query = input("Welcome to the Math Problem Recommender! Please enter your query below.\nQuery > ").strip()
    return query