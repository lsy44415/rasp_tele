import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets



params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DO015QAW"
board = BoardShim(BoardIds.CYTON_BOARD, params)
board.prepare_session()
board.start_stream()

for i in range(10):
    time.sleep(1)
    print(i)
    board.insert_marker(i + 1)
data = board.get_board_data()
board.stop_stream()
board.release_session()

print(data)
# time.sleep(10)
# # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
# data = board.get_board_data()  # get all data and remove it from internal buffer
# board.stop_stream()
# board.release_session()
