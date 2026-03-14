"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum


class DevicePlatform(str, Enum):
    IOS = "ios"
    """Apple iOS (iPhone, iPad)"""
    ANDROID = "android"
    """Google Android"""
    MACOS = "macos"
    """Apple macOS"""
    LINUX = "linux"
    """Linux (server, edge, embedded)"""
    WINDOWS = "windows"
    """Microsoft Windows"""
    BROWSER = "browser"
    """Web browser (Chrome, Safari, Firefox, etc.)"""
