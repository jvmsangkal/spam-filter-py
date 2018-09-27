import re
import email


class Document(object):
    def __init__(self, label, raw_content):
        self._label = label
        self._raw_content = raw_content
        self._message = email.message_from_string(self._raw_content)
        self._get_tokens()

    @property
    def label(self):
        return self._label

    @property
    def raw_content(self):
        return self._raw_content

    @property
    def message(self):
        return self._message

    @property
    def tokens(self):
        return self._tokens

    def _get_tokens(self):
        self._tokens = self._get_subject_tokens() + self._get_body_tokens()

    def _clean_string(self, string):
        clean_text = re.sub(r'\r(?!\n)', '\r\n', string)
        clean_text = re.sub(r'<.*?>', '', string)
        return clean_text

    def _find_match(self, string):
        if not string:
            return []

        return re.findall(r'[a-zA-Z]+', string)

    def _get_body_tokens(self):
        b = self.message
        body = ''

        if b.is_multipart():
            for part in b.walk():

                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))

                if part.get_filename():
                    continue

                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    body = part.get_payload()
                    break
        else:
            body = b.get_payload()

        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='ignore')

        body = body.lower()
        body = self._clean_string(body)

        return self._find_match(body)

    def _get_subject_tokens(self):
        return self._find_match(self.message.get('subject'))
