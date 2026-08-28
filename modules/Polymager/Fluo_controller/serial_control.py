from __future__ import annotations
import serial
import serial.tools.list_ports
from loguru import logger
import threading

class SerialControl:
    """
    Control class for Serial Communication with Fluo module.
    """
    ser: serial.Serial | None
    connected: bool
    lock: threading.Lock
    baudrate: int

    def __init__(self) -> None:
        self.ser = None
        self.connected = False
        self.lock = threading.Lock()
        self.baudrate = 115200

    @staticmethod
    def get_available_ports() -> list[str]:
        """Returns a list of available serial ports"""
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            logger.error(f"Error listing ports: {e}")
            return []

    def connect(self, port: str, timeout: float | int = 1) -> bool:
        """Connects to the specified serial port with fixed baudrate 115200"""
        with self.lock:
            if self.connected:
                self.disconnect()
            
            try:
                self.ser = serial.Serial(port, self.baudrate, timeout=timeout)
                self.connected = True
                logger.success(f"Connected to {port} at {self.baudrate} baud")
                return True
            except serial.SerialException as e:
                logger.error(f"Failed to connect to {port}: {e}")
                self.connected = False
                return False

    def disconnect(self) -> None:
        """Disconnects the current serial connection"""
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception as e:
                    logger.error(f"Error closing serial port: {e}")
            self.ser = None
            self.connected = False
            logger.info("Serial device disconnected")

    def send_command(self, data: str | bytes) -> bool:
        """Sends string data to the serial port"""
        with self.lock:
            if not self.connected or not self.ser:
                return False
            
            try:
                if isinstance(data, str):
                    raw_data = data.encode('utf-8')
                else:
                    raw_data = data
                
                self.ser.write(raw_data)
                return True
            except Exception as e:
                logger.error(f"Error sending data: {e}")
                return False

    def read(self) -> str | None:
        """Reads data from the serial port"""
        if not self.connected or not self.ser:
            return None
        
        try:
            # Read all available bytes
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                try:
                    return data.decode('utf-8', errors='replace')
                except (UnicodeDecodeError, ValueError):
                    return str(data)
            return None
        except Exception as e:
            logger.error(f"Error reading serial: {e}")
            self.disconnect()
            return None

