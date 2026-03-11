from modules.caller_id_extractor import CallerIDExtractor
extractor = CallerIDExtractor()
number = extractor.extract_caller_id()
print(f"Found number: {number}")
