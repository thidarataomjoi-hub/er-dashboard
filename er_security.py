from html import escape


def escape_html(value: object) -> str:
    return escape(str(value))
