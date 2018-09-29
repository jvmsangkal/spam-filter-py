from nltk.corpus import stopwords

import re
import email
import base64
import quopri


class Document(object):
    def __init__(self, label, raw_file, remove_stop_words):
        self._label = label
        self._raw_file = raw_file

        self._message = email.message_from_file(raw_file)

        self._get_tokens(remove_stop_words)

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

    def _get_tokens(self, remove_stop_words):
        tokens = self._get_subject_tokens() + self._get_body_tokens()

        stop_words = []
        if remove_stop_words:
            stop_words = set(stopwords.words('english'))

        self._tokens = [token for token in tokens if token not in stop_words]

    def _clean_string(self, string):
        clean_text = re.sub(r'<.*?>', '', string)
        return clean_text

    def _find_match(self, string):
        if not string:
            return []

        s = re.split(r'\s|\,|\.', string)
        r = re.compile(r'^[a-zA-Z]{2,}$')
        return list(filter(r.match, s))

    def _get_body_tokens(self):
        payloads = self._get_payloads(self.message)

        tokens = []
        for payload in payloads:
            tokens += self._find_match(self._clean_string(payload.lower()))

        return tokens

    def _get_payloads(self, message):
        payloads = []

        if message.is_multipart():
            for part in message.walk():
                sub_part_payload = part.get_payload()
                if isinstance(sub_part_payload, list):
                    for s in sub_part_payload:
                        payloads += self._get_payloads(s)
                else:
                    self._get_payloads(part)
        else:
            content_type = message.get_content_type().lower()
            content_disposition = str(message.get('Content-Disposition'))
            transfer_encoding = message.get(
                'content-transfer-encoding', '').lower()

            if 'text' in content_type and \
               'attachment' not in content_disposition:
                s = message.get_payload().strip()

                if transfer_encoding == 'base64' and ' ' not in s:
                    s = base64.b64decode(s).decode('latin-1')

                if transfer_encoding == 'quoted-printable':
                    s = s.encode('ascii', errors='ignore').decode()
                    s = quopri.decodestring(s).decode('latin-1')

                payloads.append(s)

        return payloads

    def _get_subject_tokens(self):
        subject = self.message.get('subject')

        if not subject:
            return []

        subject = self.message.get('subject').lower()
        subject = self._clean_string(subject)

        return self._find_match(subject)
