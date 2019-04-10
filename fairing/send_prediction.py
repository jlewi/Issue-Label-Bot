import argparse

def send_request(*args, **kwargs):
  # We don't use util.run because that ends up including the access token
  # in the logs
  token = subprocess.check_output(["gcloud", "auth", "print-access-token"])
  if six.PY3 and hasattr(token, "decode"):
    token = token.decode()
  token = token.strip()

  headers = {
    "Authorization": "Bearer " + token,
  }

  if "headers" not in kwargs:
    kwargs["headers"] = {}

  kwargs["headers"].update(headers)

  r = requests.post(*args, **kwargs)

  return r

if __name__ == "__main__":
  argparse.ArgumentParser()
  # We proxy the request through the APIServer so that we can connect
  # from outside the cluster.
  url = ("https://{master}/api/v1/namespaces/{namespace}/services/{service}:8500"
         "/proxy/v1/models/mnist:predict").format(
           master=master, namespace=namespace, service=service)
  logging.info("Request: %s", url)
  r = send_request(url, json=instances, verify=False)

  if r.status_code != requests.codes.OK:
    msg = "Request to {0} exited with status code: {1} and content: {2}".format(
      url, r.status_code, r.content)
    logging.error(msg)
    raise RuntimeError(msg)

  content = r.content
  if six.PY3 and hasattr(content, "decode"):
    content = content.decode()
  result = json.loads(content)
  assert len(result["predictions"]) == 1
  predictions = result["predictions"][0]