import usb.core
import usb.util

# Find all connected USB devices
devices = usb.core.find(find_all=True)

print("Connected USB Devices (WinUSB/libusb accessible):")
found = False
for d in devices:
    try:
        print(f"Device: {d.idVendor:04x}:{d.idProduct:04x} - {usb.util.get_string(d, d.iManufacturer)} {usb.util.get_string(d, d.iProduct)}")
    except Exception as e:
        print(f"Device: {d.idVendor:04x}:{d.idProduct:04x} - (Could not read strings, maybe no WinUSB driver)")
    found = True

if not found:
    print("No devices found. Libusb might not be configured correctly.")
