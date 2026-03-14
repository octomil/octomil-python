"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class NetworkType(str, Enum):
    WIFI = "wifi"
    """Connected via Wi-Fi"""
    CELLULAR = "cellular"
    """Connected via cellular data"""
    ETHERNET = "ethernet"
    """Connected via wired Ethernet"""
    NONE = "none"
    """No network connection"""
