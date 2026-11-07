from __future__ import annotations

import threading
from typing import Callable, Optional
import serial
import serial.tools.list_ports
from loguru import logger


class SerialControl:
    """
    Control class for Serial Communication with Fluo module.
    """
    ser: serial.Serial | None
    connected: bool
    lock: threading.RLock
    baudrate: int
    on_disconnect: Optional[Callable[[], None]]

    def __init__(self) -> None:
        self.ser = None
        self.connected = False
        self.lock = threading.RLock()
        self.baudrate = 115200
        self.on_disconnect = None

    @staticmethod
    def get_available_ports() -> list[str]:
        """Returns a list of available serial ports"""
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            logger.error(f"Error listing ports: {e}")
            return []

    def connect(self, port: str, timeout: float | int = 1, write_timeout: float | int = 1.0) -> bool:
        """Connects to the specified serial port with fixed baudrate 115200"""
        with self.lock:
            if self.connected or self.ser is not None:
                self.disconnect()
            
            try:
                self.ser = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    timeout=timeout,
                    write_timeout=write_timeout,
                )
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
                self.connected = True
                logger.success(f"Connected to {port} at {self.baudrate} baud")
                return True
            except (serial.SerialException, Exception) as e:
                logger.error(f"Failed to connect to {port}: {e}")
                self.ser = None
                self.connected = False
                return False

    def disconnect(self) -> None:
        """Disconnects the current serial connection safely without blocking indefinitely"""
        with self.lock:
            ser_to_close = self.ser
            self.ser = None
            was_connected = self.connected
            self.connected = False

        if ser_to_close is not None:
            def _do_close(s: serial.Serial) -> None:
                try:
                    if s.is_open:
                        try:
                            s.reset_input_buffer()
                        except Exception:
                            pass
                        try:
                            s.reset_output_buffer()
                        except Exception:
                            pass
                        s.close()
                except Exception as e:
                    logger.error(f"Error closing serial port: {e}")

            close_thread = threading.Thread(target=_do_close, args=(ser_to_close,), daemon=True)
            close_thread.start()
            close_thread.join(timeout=1.5)
            if close_thread.is_alive():
                logger.warning("Serial port close timed out (likely driver issue); continuing in background.")

        if was_connected:
            logger.info("Serial device disconnected")
            if self.on_disconnect is not None:
                try:
                    self.on_disconnect()
                except Exception as cb_err:
                    logger.error(f"Error in on_disconnect callback: {cb_err}")

    def send_command(self, data: str | bytes) -> bool:
        """Sends string data to the serial port"""
        with self.lock:
            if not self.connected or not self.ser or not self.ser.is_open:
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
        should_disconnect = False
        with self.lock:
            if not self.connected or not self.ser or not self.ser.is_open:
                return None
            
            try:
                waiting = self.ser.in_waiting
                if waiting > 0:
                    data = self.ser.read(waiting)
                    try:
                        return data.decode('utf-8', errors='replace')
                    except (UnicodeDecodeError, ValueError):
                        return str(data)
                return None
            except Exception as e:
                logger.error(f"Error reading serial: {e}")
                should_disconnect = True

        if should_disconnect:
            self.disconnect()
        return None


