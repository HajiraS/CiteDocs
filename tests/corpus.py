"""A tiny two-document corpus shared by the fixtures and tests."""

DHCP_URL = "https://docs.example.com/dhcp-leases"
DNS_URL = "https://docs.example.com/dns-caching"
DHCP_TITLE = "DHCP Leases Overview"
DNS_TITLE = "DNS Caching Overview"

# Each document fits in one chunk at the test chunk size, so its chunk id ends in -000.
DHCP_ID = "dhcp-leases-000"
DNS_ID = "dns-caching-000"

CHUNK_TOKENS = 200
CHUNK_OVERLAP = 20

DHCP_QUESTION = "How often must a DHCP client renew its lease?"
DNS_QUESTION = "What does the TTL do in DNS caching?"
OFF_TOPIC_QUESTION = "Who won the cricket world cup?"

DOCS = {
    "dhcp-leases": (
        DHCP_URL,
        DHCP_TITLE,
        "A DHCP server assigns an IP address to a client for a limited period called a lease. "
        "The client must renew the lease before it expires, normally when half of the lease "
        "time has passed. If renewal fails, the client asks any available DHCP server for a "
        "new address. Reservations bind an IP address to a client hardware address.",
    ),
    "dns-caching": (
        DNS_URL,
        DNS_TITLE,
        "A caching DNS resolver stores answers from authoritative name servers so repeated "
        "queries are answered locally. Each record has a time to live, or TTL, that says how "
        "long the answer may be reused. When the TTL expires the resolver discards the entry "
        "and asks the authoritative server again. Administrators can flush the cache after a "
        "record changes.",
    ),
}
