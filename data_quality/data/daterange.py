from datetime import datetime


class DateRange(object):

    def __init__(self, from_date, to_date, date_format='D-%d/%m/%y'):
        self._from_date = datetime.strptime(from_date, date_format)
        self._to_date = datetime.strptime(to_date, date_format)

    @property
    def from_date(self):
        return self._from_date

    @property
    def to_date(self):
        return self._to_date
