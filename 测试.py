import urllib.request

response = urllib.request.urlopen('http://www.example.com')
html = response.read()
print(html)