from pymol import cmd
import time

colors = ['0xf3faec', '0xf7fcf0', '0xe4f4de', '0xf3faec', '0xe9f6e3', '0xcbeac4', '0xd8f0d2', '0xd8f0d2', '0xf7fcf0', '0xf7fcf0', '0xf7fcf0', '0x7dcdc3', '0x9fdab8', '0xd0edca', '0xebf7e5', '0x80cec2', '0xddf2d7', '0xd8f0d2', '0xddf2d7', '0x57b8d0', '0xd9f0d3', '0xf4fbed', '0xf7fcf0', '0xd0edca', '0xd4eece', '0xf3faec', '0xebf7e5', '0xd5efcf', '0xf1faeb', '0xd5efcf', '0xe9f6e3', '0xf7fcf0', '0xe9f6e3', '0x9cd9b9', '0xebf7e5', '0xf6fbef', '0xbce5be', '0xeaf7e4', '0xa2dbb7', '0xf3faec', '0xe1f3dc', '0xf6fbef', '0xd9f0d3', '0xf6fbef', '0xf7fcf0', '0xedf8e7', '0xf4fbed', '0xb5e2bb', '0xf7fcf0', '0x80cec2', '0xe4f5df', '0xe7f6e2', '0xf3faec', '0xceecc7', '0xedf8e7', '0xdff3da', '0xdaf1d5', '0x88d1c0', '0xd1edcb', '0xe1f3dc', '0xbfe6bf', '0xcfecc8', '0xccebc6', '0x084081', '0xf1faeb', '0xcbeac4', '0xaadeb6', '0xa5dcb6', '0xeef9e8', '0xd7efd1', '0xaadeb6', '0xceecc7', '0xeef9e8', '0x99d7ba', '0xe1f3dc', '0xf7fcf0', '0xf6fbef', '0xddf2d7', '0xd7efd1', '0xf7fcf0', '0xd4eece']
print(colors)
index = 135
for c in colors:
    cmd.do('color {0}, resi {1}'.format(c,index))
    index+=1
    time.sleep(.001)