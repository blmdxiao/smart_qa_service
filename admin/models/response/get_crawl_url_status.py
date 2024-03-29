

class URL:
    length: int
    status: str
    url: str
    url_id: str

    def __init__(self, length: int, status: str, url: str, url_id: str) -> None:
        self.length = length
        self.status = status
        self.url = url
        self.url_id = url_id


class UrlStatusData:
    """1-crawling"""
    status: int
    urls: []

    def __init__(self, status: int, urls: list) -> None:
        self.status = status
        self.urls = urls


class CommonResponse:
    """Request"""
    data: UrlStatusData
    message: str
    retcode: int

    def __init__(self, data: UrlStatusData, message: str, retcode: int) -> None:
        self.data = data
        self.message = message
        self.retcode = retcode