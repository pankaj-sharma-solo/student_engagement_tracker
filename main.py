from extractor.Extractor import Extractor

if __name__ == '__main__':
    with Extractor() as fe:
        fe.facialFeatureExtractor()
