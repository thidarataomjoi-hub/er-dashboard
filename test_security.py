import unittest

from er_security import escape_html


class HtmlEscapingTests(unittest.TestCase):
    def test_escapes_markup_and_attribute_quotes(self) -> None:
        self.assertEqual(
            escape_html('<img src=x onerror="alert(1)"> &'),
            "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp;",
        )


if __name__ == "__main__":
    unittest.main()
