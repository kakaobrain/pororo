"""Test OCR module"""

import unittest

from pororo import Pororo


class PororoOCRTester(unittest.TestCase):

    def test_modules(self):
        img_path = "https://kixxman.com/files/attach/images/140/554/007/1e57a1b02405d5d955030ced868375c2.jpg"
        ocr = Pororo(task="ocr", lang="ko")
        ocr = ocr(img_path)
        self.assertIsInstance(ocr, list)

        ocr = Pororo(task="ocr", lang="ko")
        ocr = ocr(img_path, detail=True)
        self.assertIsInstance(ocr, dict)


if __name__ == "__main__":
    unittest.main()
