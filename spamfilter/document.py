import re
import mailparser


class Document(object):
    def __init__(self, label, raw_file):
        self._label = label
        self._raw_file = raw_file
        self._message = mailparser.parse_from_string(self.raw_file.read())
        self._get_tokens()

    @property
    def label(self):
        return self._label

    @property
    def raw_file(self):
        return self._raw_file

    @property
    def message(self):
        return self._message

    @property
    def tokens(self):
        return self._tokens

    def _get_tokens(self):
        self._tokens = self._get_subject_tokens() + self._get_body_tokens()

    def _clean_string(self, string):
        clean_text = re.sub(r'<.*?>', '', string)
        return clean_text

    def _find_match(self, string):
        if not string:
            return []

        return re.findall(r'[a-zA-Z]+', string)

    def _get_body_tokens(self):
        tokens = []

        for text in self.message.text_plain:
            tokens += self._find_match(self._clean_string(text.lower()))

        for text in self.message.text_html:
            self._find_match(self._clean_string(text.lower()))

        return tokens

    def _get_subject_tokens(self):
        subject = self.message.subject.lower()
        subject = self._clean_string(subject)

        return self._find_match(subject)
