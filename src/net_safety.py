"""
Which network addresses may a server-side fetch connect to?

`ipaddress.ip_address(x).is_global` is the right first test but it has gaps that matter for SSRF (checked on Python 3.12):

  * 64:ff9b::a9fe:a9fe and 64:ff9b::7f00:1 (the NAT64 forms of 169.254.169.254, the cloud metadata service, and of 127.0.0.1) are
    reported as global. On a network with NAT64 an attacker-controlled DNS answer reaches the metadata service through them.
  * multicast addresses are reported as global.
  * 6to4 (2002::/16), Teredo (2001::/32) and IPv4-compatible (::/96) addresses hide an IPv4 address inside an IPv6 one.

is_public_ip() closes those. resolve_public() resolves a name and returns its addresses only if EVERY one of them is public: a name
that also resolves to a private address is refused, because the connection could be made to either.
"""
from __future__ import annotations

import ipaddress
import socket
from typing import List, Union

_NAT64 = ipaddress.ip_network("64:ff9b::/96")                    # RFC 6052: an IPv4 address in the last 32 bits
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")            # RFC 8215: local use, never public
_TEREDO = ipaddress.ip_network("2001::/32")
_SIX_TO_FOUR = ipaddress.ip_network("2002::/16")
_V4_COMPATIBLE = ipaddress.ip_network("::/96")                   # deprecated, and ::7f00:1 would be 127.0.0.1

Address = Union[str, ipaddress.IPv4Address, ipaddress.IPv6Address]


def is_public_ip(value: Address) -> bool:
    """True only for a globally routable unicast address that does not hide a non-public IPv4 address."""
    try:
        ip = value if isinstance(value, (ipaddress.IPv4Address, ipaddress.IPv6Address)) else ipaddress.ip_address(str(value).strip("[]"))
    except ValueError:
        return False
    if ip.version == 6:
        if ip.ipv4_mapped is not None:                           # ::ffff:10.0.0.1
            return is_public_ip(ip.ipv4_mapped)
        if ip in _NAT64:                                         # 64:ff9b::a9fe:a9fe is 169.254.169.254
            return is_public_ip(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip in _NAT64_LOCAL or ip in _TEREDO or ip in _SIX_TO_FOUR or ip in _V4_COMPATIBLE:
            return False
    if ip.is_multicast:
        return False
    return ip.is_global


def resolve_public(host: str) -> List[str]:
    """The addresses `host` resolves to, if all of them are public; [] when it does not resolve or any one of them is not."""
    host = (host or "").strip().strip("[]").rstrip(".")
    if not host:
        return []
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    found: List[str] = []
    for info in infos:
        address = info[4][0]
        if not is_public_ip(address):
            return []
        if address not in found:
            found.append(address)
    return found
