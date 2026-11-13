"""Which addresses a server-side fetch may connect to: the SSRF address policy, shared by the web reader and the document import."""
import ipaddress
import socket

import pytest

import src.text_processing_agent as tpa
from src.net_safety import is_public_ip, resolve_public

PUBLIC = ["93.184.216.34", "8.8.8.8", "1.1.1.1", "2606:2800:220:1:248:1893:25c8:1946", "2001:4860:4860::8888", "64:ff9b::808:808"]    # last: NAT64 of 8.8.8.8
PRIVATE = [
    "127.0.0.1", "127.1.2.3", "10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0", "255.255.255.255",
    "192.0.0.1", "198.18.0.1", "192.0.2.1", "198.51.100.1", "203.0.113.1", "240.0.0.1",
    "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "2001:db8::1",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254",
    "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b::a00:1", "64:ff9b:1::1",       # NAT64 forms of 127.0.0.1, 169.254.169.254, 10.0.0.1; local-use NAT64
    "2002:7f00:1::1", "2002:a9fe:a9fe::1", "2002:808:808::1",                         # 6to4, even for a public embedded address
    "2001:0:4136:e378:8000:63bf:3fff:fdd2",                                           # Teredo
    "::7f00:1", "::a9fe:a9fe",                                                        # IPv4-compatible
    "224.0.0.1", "239.255.255.250", "ff02::1", "ff0e::1",                             # multicast
]


@pytest.mark.parametrize("address", PUBLIC)
def test_public_addresses_are_allowed(address):
    assert is_public_ip(address)


@pytest.mark.parametrize("address", PRIVATE)
def test_private_reserved_and_disguised_addresses_are_refused(address):
    assert not is_public_ip(address)
    assert not is_public_ip(ipaddress.ip_address(address))


@pytest.mark.parametrize("junk", ["", "not an ip", "999.1.1.1", "1.2.3", "0x7f000001", "127.0.0.1/8", None])
def test_anything_that_is_not_an_address_is_refused(junk):
    assert not is_public_ip(junk)


def test_ipaddress_alone_would_have_allowed_the_nat64_metadata_address():
    """The reason this module exists: is_global says yes to the NAT64 form of the cloud metadata address."""
    assert ipaddress.ip_address("64:ff9b::a9fe:a9fe").is_global and not is_public_ip("64:ff9b::a9fe:a9fe")
    assert ipaddress.ip_address("224.0.0.1").is_global and not is_public_ip("224.0.0.1")


def fake_dns(monkeypatch, answers):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [(socket.AF_INET6 if ":" in x else socket.AF_INET, socket.SOCK_STREAM, 6, "", (x, 0)) for x in answers])


def test_a_name_is_refused_when_any_of_its_addresses_is_not_public(monkeypatch):
    fake_dns(monkeypatch, ["93.184.216.34", "10.0.0.5"])
    assert resolve_public("mixed.example") == []
    fake_dns(monkeypatch, ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"])
    assert resolve_public("ok.example") == ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]
    fake_dns(monkeypatch, ["64:ff9b::a9fe:a9fe"])
    assert resolve_public("dns64.example") == []


def test_a_name_that_does_not_resolve_is_refused(monkeypatch):
    def boom(*a, **k):
        raise socket.gaierror("nope")
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert resolve_public("nowhere.invalid") == [] and resolve_public("") == [] and resolve_public(None) == []


def test_the_existing_web_reader_uses_the_same_policy(monkeypatch):
    fake_dns(monkeypatch, ["64:ff9b::a9fe:a9fe"])
    assert tpa._resolve_public_addresses("nat64-metadata-test.example") == ()
    fake_dns(monkeypatch, ["224.0.0.1"])
    assert tpa._resolve_public_addresses("multicast-test.example") == ()
    fake_dns(monkeypatch, ["93.184.216.34"])
    assert tpa._resolve_public_addresses("public-test.example") == ("93.184.216.34",)
