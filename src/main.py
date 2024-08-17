from parser.statement_parser import StatementParser

bs_parse = StatementParser(show_detections=True, cell_padding=20)
bs_parse.bankstatement2csv(pdf='/Users/farhanishraq/Downloads/Llama/BankAIAgent/statements/example.pdf')