from apple_health_parser.utils.parser import Parser


parse = Parser(r"export.zip")  # or an absolute path
parse.export(r"files")
