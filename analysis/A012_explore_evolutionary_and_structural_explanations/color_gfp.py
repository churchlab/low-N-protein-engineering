from pymol import cmd
import time

colors = ['#d1dcba', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#d1dcba', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d4ddbd', '#cfdbb9', '#d7dfc0', '#d4ddbd', '#d4ddbd', '#ccdab6', '#d1dcba', '#d7dfc0', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#d7dfc0', '#d1dcba', '#d7dfc0', '#d1dcba', '#d4ddbd', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#c2d6af', '#d7dfc0', '#d7dfc0', '#d4ddbd', '#d1dcba', '#d7dfc0', '#d7dfc0', '#cfdbb9', '#d4ddbd', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d1dcba', '#d4ddbd', '#c9d9b4', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d1dcba', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#ccdab6', '#d4ddbd', '#c1d6ad', '#d7dfc0', '#c9d9b4', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#cfdbb9', '#d4ddbd', '#cfdbb9', '#d7dfc0', '#d7dfc0', '#d7dfc0']

colors = ['#d1dcbaff', '#d7dfc0ff', '#d4ddbdff', '#d7dfc0ff', '#d1dcbaff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d4ddbdff', '#cfdbb9ff', '#d7dfc0ff', '#d4ddbdff', '#d4ddbdff', '#ccdab6ff', '#d1dcbaff', '#d7dfc0ff', '#d7dfc0ff', '#d4ddbdff', '#d7dfc0ff', '#d7dfc0ff', '#d1dcbaff', '#d7dfc0ff', '#d1dcbaff', '#d4ddbdff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#c2d6afff', '#d7dfc0ff', '#d7dfc0ff', '#d4ddbdff', '#d1dcbaff', '#d7dfc0ff', '#d7dfc0ff', '#cfdbb9ff', '#d4ddbdff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d1dcbaff', '#d4ddbdff', '#c9d9b4ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d4ddbdff', '#d7dfc0ff', '#d4ddbdff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#d1dcbaff', '#d7dfc0ff', '#d4ddbdff', '#d7dfc0ff', '#ccdab6ff', '#d4ddbdff', '#c1d6adff', '#d7dfc0ff', '#c9d9b4ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff', '#cfdbb9ff', '#d4ddbdff', '#cfdbb9ff', '#d7dfc0ff', '#d7dfc0ff', '#d7dfc0ff']

colors = ['#000000','#FF0000'] * 20
colors = ['#00483d', '#003f34', '#004439', '#003f34', '#00483d', '#003f34', '#003f34', '#003f34', '#004439', '#004c42', '#003f34', '#004439', '#004439', '#005046', '#00483d', '#003f34', '#003f34', '#004439', '#003f34', '#003f34', '#00483d', '#003f34', '#00483d', '#004439', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#015d54', '#003f34', '#003f34', '#004439', '#00483d', '#003f34', '#003f34', '#004c42', '#004439', '#003f34', '#003f34', '#003f34', '#003f34', '#00483d', '#004439', '#01554b', '#003f34', '#003f34', '#003f34', '#004439', '#003f34', '#004439', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#003f34', '#00483d', '#003f34', '#004439', '#003f34', '#005046', '#004439', '#016058', '#003f34', '#01554b', '#003f34', '#003f34', '#003f34', '#004c42', '#004439', '#004c42', '#003f34', '#003f34', '#003f34']

colors = ['#d2ddbc', '#d7dfc0', '#d2ddbc', '#d7dfc0', '#d2ddbc', '#d7dfc0', '#d5debe', '#d5debe', '#d4ddbd', '#cedbb7', '#d7dfc0', '#d4ddbd', '#d2ddbc', '#cbdab5', '#d2ddbc', '#d5debe', '#d7dfc0', '#d2ddbc', '#d7dfc0', '#d5debe', '#d2ddbc', '#d7dfc0', '#d2ddbc', '#d4ddbd', '#d7dfc0', '#d5debe', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d5debe', '#d7dfc0', '#d5debe', '#d5debe', '#d7dfc0', '#d5debe', '#c1d6ad', '#d7dfc0', '#d7dfc0', '#d2ddbc', '#d1dcba', '#d7dfc0', '#d7dfc0', '#cfdbb9', '#d4ddbd', '#d7dfc0', '#d7dfc0', '#d5debe', '#d5debe', '#d1dcba', '#d2ddbc', '#c7d8b2', '#d7dfc0', '#d7dfc0', '#d5debe', '#d5debe', '#d7dfc0', '#d5debe', '#d5debe', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#d1dcba', '#d7dfc0', '#d4ddbd', '#d7dfc0', '#ccdab6', '#d4ddbd', '#c1d6ad', '#d5debe', '#c7d8b2', '#d7dfc0', '#d7dfc0', '#d7dfc0', '#ccdab6', '#d2ddbc', '#cfdbb9', '#d7dfc0', '#d5debe', '#d7dfc0']
colors = ['#46085c', '#440154', '#46075a', '#440154', '#46085c', '#440154', '#440256', '#440256', '#450559', '#470d60', '#440154', '#450559', '#46075a', '#471164', '#46085c', '#450457', '#440154', '#46075a', '#440154', '#440256', '#46085c', '#440154', '#46085c', '#450559', '#440154', '#440256', '#440154', '#440154', '#440154', '#440256', '#440154', '#440256', '#440256', '#440154', '#440256', '#481d6f', '#440154', '#440154', '#46075a', '#460a5d', '#440154', '#440154', '#460b5e', '#450559', '#440154', '#440154', '#440256', '#440256', '#460a5d', '#46075a', '#481668', '#440154', '#440154', '#440256', '#450457', '#440154', '#450457', '#440256', '#440154', '#440154', '#440154', '#440154', '#440154', '#460a5d', '#440154', '#450559', '#440154', '#471063', '#450559', '#481d6f', '#440256', '#481668', '#440154', '#440154', '#440154', '#471063', '#46075a', '#460b5e', '#440154', '#440256', '#440154']
colors = ['#21918c', '#21918c', '#440256', '#440154', '#21918c', '#440256', '#440154', '#21918c', '#21918c', '#21918c', '#440256', '#21918c', '#21918c', '#440256', '#21918c', '#440154', '#21918c', '#21918c', '#440154', '#440154', '#21918c', '#440154', '#440256', '#440154', '#440154', '#21918c', '#440154', '#21918c', '#21918c', '#21918c', '#440256', '#440154', '#440154', '#21918c', '#440256', '#21918c', '#21918c', '#440154', '#21918c', '#440256', '#21918c', '#21918c', '#21918c', '#21918c', '#440154', '#440256', '#440154', '#21918c', '#440154', '#21918c', '#440154', '#440256', '#21918c', '#440256', '#21918c', '#440256', '#21918c', '#440256', '#440154', '#440256', '#440256', '#440154', '#21918c', '#440256', '#440256', '#440154', '#440154', '#440154', '#440154', '#440154', '#440256', '#440256', '#440154', '#440256', '#21918c', '#21918c', '#440154', '#440256', '#21918c', '#21918c', '#440154']
colors = ['#440154ff', '#440256ff', '#440256ff', '#21918cff', '#440154ff', '#21918cff', '#21918cff', '#440256ff', '#440256ff', '#440256ff', '#440256ff', '#21918cff', '#21918cff', '#21918cff', '#440256ff', '#21918cff', '#440256ff', '#440256ff', '#440154ff', '#440154ff', '#440154ff', '#440256ff', '#440256ff', '#440154ff', '#21918cff', '#440256ff', '#440256ff', '#440256ff', '#440154ff', '#21918cff', '#21918cff', '#440256ff', '#440154ff', '#21918cff', '#440256ff', '#440256ff', '#440256ff', '#440154ff', '#440154ff', '#440154ff', '#440256ff', '#440256ff', '#440154ff', '#21918cff', '#21918cff', '#440154ff', '#21918cff', '#21918cff', '#440256ff', '#440256ff', '#440154ff', '#21918cff', '#440154ff', '#440256ff', '#440256ff', '#440154ff', '#440256ff', '#440256ff', '#440256ff', '#440154ff', '#440256ff', '#21918cff', '#440256ff', '#440256ff', '#440154ff', '#21918cff', '#440154ff', '#21918cff', '#440154ff', '#440154ff', '#440256ff', '#440154ff', '#440154ff', '#21918cff', '#440154ff', '#440256ff', '#21918cff', '#21918cff', '#440256ff', '#440154ff', '#440154ff']
colors = ['#fed976', '#e2191c', '#fd8c3c', '#fed976', '#fed976', '#e2191c', '#e2191c', '#fed976', '#e2191c', '#fed976', '#fed976', '#ffffcc', '#fd8c3c', '#ffffcc', '#fffecb', '#fd8c3c', '#fed976', '#e2191c', '#fffecb', '#fd8c3c', '#e2191c', '#fed976', '#e2191c', '#e2191c', '#ffffcc', '#e2191c', '#fffecb', '#ffffcc', '#fffecb', '#ffffcc', '#fd8c3c', '#fd8c3c', '#ffffcc', '#fed976', '#fffecb', '#fd8c3c', '#fed976', '#ffffcc', '#fd8c3c', '#fffecb', '#e2191c', '#e2191c', '#e2191c', '#fffecb', '#e2191c', '#fd8c3c', '#ffffcc', '#fed976', '#fed976', '#fffecb', '#fed976', '#ffffcc', '#fed976', '#fed976', '#fed976', '#fed976', '#ffffcc', '#fed976', '#ffffcc', '#fffecb', '#fed976', '#fd8c3c', '#fffecb', '#ffffcc', '#ffffcc', '#fffecb', '#fed976', '#e2191c', '#fed976', '#fed976', '#e2191c', '#fffecb', '#fed976', '#e2191c', '#fd8c3c', '#e2191c', '#fffecb', '#fed976', '#e2191c', '#fed976', '#ffffcc']
colors = ['#fffcc5', '#ffffcc', '#fffdc6', '#ffffcc', '#fffcc5', '#ffffcc', '#fffecb', '#fffecb', '#fffdc8', '#fffac1', '#ffffcc', '#fffdc8', '#fffdc6', '#fff9bd', '#fffcc5', '#fffec9', '#ffffcc', '#fffdc6', '#ffffcc', '#fffecb', '#fffcc5', '#ffffcc', '#fffcc5', '#fffdc8', '#ffffcc', '#fffecb', '#ffffcc', '#ffffcc', '#ffffcc', '#fffecb', '#ffffcc', '#fffecb', '#fffecb', '#ffffcc', '#fffecb', '#fff4b0', '#ffffcc', '#ffffcc', '#fffdc6', '#fffcc4', '#ffffcc', '#ffffcc', '#fffbc2', '#fffdc8', '#ffffcc', '#ffffcc', '#fffecb', '#fffecb', '#fffcc4', '#fffdc6', '#fff7b9', '#ffffcc', '#ffffcc', '#fffecb', '#fffec9', '#ffffcc', '#fffec9', '#fffecb', '#ffffcc', '#ffffcc', '#ffffcc', '#ffffcc', '#ffffcc', '#fffcc4', '#ffffcc', '#fffdc8', '#ffffcc', '#fff9be', '#fffdc8', '#fff4b0', '#fffecb', '#fff7b9', '#ffffcc', '#ffffcc', '#ffffcc', '#fff9be', '#fffdc6', '#fffbc2', '#ffffcc', '#fffecb', '#ffffcc']
colors = ['#fed572', '#ffffcc', '#fedd7f', '#ffffcc', '#fed976', '#ffffcc', '#fff1ab', '#fff8ba', '#ffe48c', '#fea948', '#ffffcc', '#ffe997', '#fedc7c', '#fd7435', '#fed976', '#fff1a9', '#fffdc8', '#fedc7c', '#ffffcc', '#fff7b7', '#fed976', '#ffffcc', '#fed572', '#ffe794', '#ffffcc', '#fff1ab', '#ffffcc', '#ffffcc', '#ffffcc', '#fff7b7', '#fffac0', '#fff6b6', '#fff7b7', '#fff9be', '#fff5b3', '#8b0026', '#fffecb', '#ffffcc', '#fee084', '#fec35e', '#ffffcc', '#fffecb', '#feb44e', '#ffe997', '#ffffcc', '#ffffcc', '#fff3ae', '#fff1ab', '#fecc68', '#fee187', '#ed3022', '#fffbc2', '#ffffcc', '#fff4b2', '#ffeda0', '#ffffcc', '#ffec9f', '#fff5b3', '#fffcc4', '#ffffcc', '#fff9be', '#ffffcc', '#fffecb', '#fec965', '#ffffcc', '#ffe48c', '#ffffcc', '#fd8c3c', '#ffe997', '#800026', '#fff8ba', '#ef3323', '#fff9be', '#fffecb', '#ffffcc', '#fd913e', '#fedd7f', '#feb24c', '#fff8bb', '#fff5b3', '#fffdc6']
colors = ['#fee08a', '#ffffe5', '#fee79a', '#ffffe5', '#fee390', '#ffffe5', '#fff9c6', '#fffcd4', '#ffeea8', '#feba46', '#ffffe5', '#fff3b2', '#fee697', '#f78921', '#fee390', '#fff9c5', '#fffee1', '#fee697', '#ffffe5', '#fffbd2', '#fee390', '#ffffe5', '#fee08a', '#fff1b0', '#ffffe5', '#fff9c6', '#ffffe5', '#ffffe5', '#ffffe5', '#fffbd2', '#fffdd9', '#fffbd0', '#fffbd2', '#fffcd8', '#ffface', '#702806', '#ffffe4', '#ffffe5', '#feeaa0', '#fed16b', '#ffffe5', '#ffffe4', '#fec652', '#fff3b2', '#ffffe5', '#ffffe5', '#fff9c9', '#fff9c6', '#fed97c', '#feeba2', '#d95b09', '#fffddc', '#ffffe5', '#fffacd', '#fff7bc', '#ffffe5', '#fff6ba', '#ffface', '#fffddd', '#ffffe5', '#fffcd8', '#ffffe5', '#ffffe4', '#fed676', '#ffffe5', '#ffeea8', '#ffffe5', '#fe9829', '#fff3b2', '#662506', '#fffcd4', '#db5d0b', '#fffcd8', '#ffffe4', '#ffffe5', '#fe9e2d', '#fee79a', '#fec34f', '#fffcd6', '#ffface', '#fffee0']
colors = ['#fdcd9c', '#fff5eb', '#fdd5ab', '#fff5eb', '#fdd0a2', '#fff5eb', '#feead5', '#ffefdf', '#fedcb9', '#fda660', '#fff5eb', '#fee1c4', '#fdd3a9', '#f87f2c', '#fdd0a2', '#fee9d4', '#fff4e8', '#fdd3a9', '#fff5eb', '#ffeedd', '#fdd0a2', '#fff5eb', '#fdcd9c', '#fee0c1', '#fff5eb', '#feead5', '#fff5eb', '#fff5eb', '#fff5eb', '#ffeedd', '#fff1e3', '#feeddc', '#ffeedd', '#fff0e2', '#feeddb', '#862a04', '#fff5ea', '#fff5eb', '#fdd7b1', '#fdbd83', '#fff5eb', '#fff5ea', '#fdb06e', '#fee1c4', '#fff5eb', '#fff5eb', '#feebd7', '#feead5', '#fdc590', '#fdd9b4', '#e35608', '#fff2e5', '#fff5eb', '#feecda', '#fee6ce', '#fff5eb', '#fee5cc', '#feeddb', '#fff2e6', '#fff5eb', '#fff0e2', '#fff5eb', '#fff5ea', '#fdc28b', '#fff5eb', '#fedcb9', '#fff5eb', '#fd8c3b', '#fee1c4', '#7f2704', '#ffefdf', '#e4580a', '#fff0e2', '#fff5ea', '#fff5eb', '#fd9141', '#fdd5ab', '#fdae6a', '#ffefe0', '#feeddb', '#fff3e7']
colors = ['#533421', '#000000', '#462d1c', '#000000', '#4f3220', '#000000', '#1e130c', '#100a06', '#3a2517', '#7f5033', '#000000', '#301e13', '#492e1d', '#ad6d46', '#4f3220', '#1f140c', '#040201', '#492e1d', '#000000', '#130c07', '#4f3220', '#000000', '#533421', '#332014', '#000000', '#1e130c', '#000000', '#000000', '#000000', '#130c07', '#0b0704', '#140c08', '#130c07', '#0c0805', '#160e09', '#ffc37c', '#010100', '#000000', '#41291a', '#654029', '#000000', '#010100', '#74492f', '#301e13', '#000000', '#000000', '#1b110b', '#1e130c', '#5b3a25', '#3f2819', '#dc8b59', '#090503', '#000000', '#170f09', '#281910', '#000000', '#291a10', '#160e09', '#070503', '#000000', '#0c0805', '#000000', '#010100', '#5f3c26', '#000000', '#3a2517', '#000000', '#9e6440', '#301e13', '#ffc77f', '#100a06', '#d98958', '#0c0805', '#010100', '#000000', '#99613e', '#462d1c', '#774b30', '#0f0906', '#160e09', '#050302']
#colors = ['#c1d6ad'] * len(colors)
colors = ['0x' + c[1:] for c in colors]
colors = ['0x89bf94', '0xd7dfc0', '0x93c496', '0xd7dfc0', '0x8cc195', '0xd7dfc0', '0xbcd4aa', '0xc7d8b2', '0xa0c99b', '0x65a88e', '0xd7dfc0', '0xabcda0', '0x93c496', '0x4c898b', '0x8cc195', '0xbad3a9', '0xd4ddbd', '0x93c496', '0xd7dfc0', '0xc7d8b2', '0x8cc195', '0xd7dfc0', '0x89bf94', '0xa7cc9f', '0xd7dfc0', '0xbcd4aa', '0xd7dfc0', '0xd7dfc0', '0xd7dfc0', '0xc7d8b2', '0xcedbb7', '0xc4d7b0', '0xc7d8b2', '0xcbdab5', '0xc2d6af', '0x2f2345', '0xd5debe', '0xd7dfc0', '0x99c799', '0x78b690', '0xd7dfc0', '0xd5debe', '0x6cae8f', '0xabcda0', '0xd7dfc0', '0xd7dfc0', '0xbdd4ab', '0xbcd4aa', '0x81bb92', '0x9bc799', '0x406381', '0xcedbb7', '0xd7dfc0', '0xc1d6ad', '0xb3d1a5', '0xd7dfc0', '0xb1d0a4', '0xc2d6af', '0xd1dcba', '0xd7dfc0', '0xcbdab5', '0xd7dfc0', '0xd5debe', '0x7eb991', '0xd7dfc0', '0xa0c99b', '0xd7dfc0', '0x52938c', '0xabcda0', '0x2c1e3e', '0xc7d8b2', '0x406582', '0xcbdab5', '0xd5debe', '0xd7dfc0', '0x55978d', '0x93c496', '0x6bad8f', '0xcbdab5', '0xc2d6af', '0xd2ddbc']
colors = ['0xe5cc8f', '0x003f34', '0xf6eacb', '0x003f34', '0xecd8a5', '0x003f34', '0x58b0a7', '0x1a7e76', '0xe9f2f1', '0x583305', '0x003f34', '0xc4e9e4', '0xf6e9c5', '0x583305', '0xecd8a5', '0x67bbb0', '0x004c42', '0xf6e9c5', '0x003f34', '0x1f827a', '0xecd8a5', '0x003f34', '0xe5cc8f', '0xceece8', '0x003f34', '0x58b0a7', '0x003f34', '0x003f34', '0x003f34', '0x1f827a', '0x01655d', '0x298b83', '0x1f827a', '0x0a6f67', '0x2f9189', '0x583305', '0x00483d', '0x003f34', '0xf5f1e4', '0xb37625', '0x003f34', '0x00483d', '0x754308', '0xc4e9e4', '0x003f34', '0x003f34', '0x4faaa1', '0x58b0a7', '0xd1a559', '0xf5f2e8', '0x583305', '0x016058', '0x003f34', '0x3b9b93', '0x92d4ca', '0x003f34', '0x9ad8ce', '0x2f9189', '0x01584f', '0x003f34', '0x0a6f67', '0x003f34', '0x00483d', '0xc89343', '0x003f34', '0xe9f2f1', '0x003f34', '0x583305', '0xc4e9e4', '0x583305', '0x1a7e76', '0x583305', '0x0a6f67', '0x00483d', '0x003f34', '0x583305', '0xf6eacb', '0x6a3d07', '0x0e726a', '0x2f9189', '0x01554b']
colors = ['0x860700', '0x000000', '0x740000', '0x000000', '0x800000', '0x000000', '0x300000', '0x1c0000', '0x5e0000', '0xce4e00', '0x000000', '0x4e0000', '0x760000', '0xff9919', '0x800000', '0x340000', '0x060000', '0x760000', '0x000000', '0x1e0000', '0x800000', '0x000000', '0x860700', '0x520000', '0x000000', '0x300000', '0x000000', '0x000000', '0x000000', '0x1e0000', '0x120000', '0x220000', '0x1e0000', '0x160000', '0x240000', '0xfffff3', '0x040000', '0x000000', '0x6a0000', '0xa42400', '0x000000', '0x040000', '0xbc3d00', '0x4e0000', '0x000000', '0x000000', '0x2e0000', '0x300000', '0x941400', '0x680000', '0xffe465', '0x100000', '0x000000', '0x280000', '0x400000', '0x000000', '0x420000', '0x240000', '0x0c0000', '0x000000', '0x160000', '0x000000', '0x040000', '0x9a1a00', '0x000000', '0x5e0000', '0x000000', '0xff8001', '0x4e0000', '0xffffff', '0x1c0000', '0xffe061', '0x160000', '0x040000', '0x000000', '0xf87800', '0x740000', '0xc04000', '0x180000', '0x240000', '0x0a0000']
colors = ['0x39558c', '0x440154', '0x3e4c8a', '0x440154', '0x3b528b', '0x440154', '0x482374', '0x481668', '0x423f85', '0x29798e', '0x440154', '0x453581', '0x3d4d8a', '0x1e9c89', '0x3b528b', '0x482576', '0x450559', '0x3d4d8a', '0x440154', '0x481769', '0x3b528b', '0x440154', '0x39558c', '0x453882', '0x440154', '0x482374', '0x440154', '0x440154', '0x440154', '0x481769', '0x470e61', '0x481a6c', '0x481769', '0x471164', '0x481b6d', '0xefe51c', '0x450457', '0x440154', '0x404688', '0x32658e', '0x440154', '0x450457', '0x2d718e', '0x453581', '0x440154', '0x440154', '0x482173', '0x482374', '0x365d8d', '0x404588', '0x42be71', '0x470d60', '0x440154', '0x481d6f', '0x472d7b', '0x440154', '0x472e7c', '0x481b6d', '0x460a5d', '0x440154', '0x471164', '0x440154', '0x450457', '0x34608d', '0x440154', '0x423f85', '0x440154', '0x21918c', '0x453581', '0xfde725', '0x481668', '0x3fbc73', '0x471164', '0x450457', '0x440154', '0x228d8d', '0x3e4c8a', '0x2c728e', '0x471365', '0x481b6d', '0x46085c']
colors = ['0x8305a7', '0x0d0887', '0x7501a8', '0x0d0887', '0x7e03a8', '0x0d0887', '0x3f049c', '0x2e0595', '0x6400a7', '0xb22b8f', '0x0d0887', '0x5801a4', '0x7701a8', '0xd6556d', '0x7e03a8', '0x43039e', '0x16078a', '0x7701a8', '0x0d0887', '0x2f0596', '0x7e03a8', '0x0d0887', '0x8305a7', '0x5b01a5', '0x0d0887', '0x3f049c', '0x0d0887', '0x0d0887', '0x0d0887', '0x2f0596', '0x240691', '0x330597', '0x2f0596', '0x280592', '0x350498', '0xf3ee27', '0x130789', '0x0d0887', '0x6e00a8', '0x9814a0', '0x0d0887', '0x130789', '0xa72197', '0x5801a4', '0x0d0887', '0x0d0887', '0x3e049c', '0x3f049c', '0x8d0ba5', '0x6c00a8', '0xf1834c', '0x220690', '0x0d0887', '0x38049a', '0x4c02a1', '0x0d0887', '0x4e02a2', '0x350498', '0x1d068e', '0x0d0887', '0x280592', '0x0d0887', '0x130789', '0x910ea3', '0x0d0887', '0x6400a7', '0x0d0887', '0xcc4778', '0x5801a4', '0xf0f921', '0x2e0595', '0xf0804e', '0x280592', '0x130789', '0x0d0887', '0xc8437b', '0x7501a8', '0xaa2395', '0x2a0593', '0x350498', '0x1b068d']
colors = ['0xfdd19b', '0xfff7ec', '0xfdd8a6', '0xfff7ec', '0xfdd49e', '0xfff7ec', '0xfeecd1', '0xfff0dc', '0xfedfb4', '0xfdb07a', '0xfff7ec', '0xfee4bf', '0xfdd7a4', '0xf77d52', '0xfdd49e', '0xfeebcf', '0xfff6e9', '0xfdd7a4', '0xfff7ec', '0xfff0db', '0xfdd49e', '0xfff7ec', '0xfdd19b', '0xfee2bc', '0xfff7ec', '0xfeecd1', '0xfff7ec', '0xfff7ec', '0xfff7ec', '0xfff0db', '0xfff3e2', '0xfeefd9', '0xfff0db', '0xfff2e0', '0xfeefd8', '0x890000', '0xfff6ea', '0xfff7ec', '0xfddbac', '0xfdc68f', '0xfff7ec', '0xfff6ea', '0xfdbc85', '0xfee4bf', '0xfff7ec', '0xfff7ec', '0xfeecd2', '0xfeecd1', '0xfdcc96', '0xfddbad', '0xe14630', '0xfff3e3', '0xfff7ec', '0xfeeed5', '0xfee8c8', '0xfff7ec', '0xfee7c7', '0xfeefd8', '0xfff4e5', '0xfff7ec', '0xfff2e0', '0xfff7ec', '0xfff6ea', '0xfdca93', '0xfff7ec', '0xfedfb4', '0xfff7ec', '0xfc8c59', '0xfee4bf', '0x7f0000', '0xfff0dc', '0xe24933', '0xfff2e0', '0xfff6ea', '0xfff7ec', '0xfc925e', '0xfdd8a6', '0xfdba83', '0xfff1de', '0xfeefd8', '0xfff5e6']
colors = ['0xc0e6b5', '0xffffd9', '0xceecb3', '0xffffd9', '0xc6e9b4', '0xffffd9', '0xf1fabb', '0xf7fcc7', '0xdbf1b2', '0x71c8bd', '0xffffd9', '0xe5f5b2', '0xcdebb4', '0x33a7c2', '0xc6e9b4', '0xf0f9b8', '0xfdfed5', '0xcdebb4', '0xffffd9', '0xf7fcc6', '0xc6e9b4', '0xffffd9', '0xc0e6b5', '0xe2f4b2', '0xffffd9', '0xf1fabb', '0xffffd9', '0xffffd9', '0xffffd9', '0xf7fcc6', '0xfafdce', '0xf5fbc4', '0xf7fcc6', '0xf9fdcb', '0xf5fbc2', '0x0d2163', '0xfeffd6', '0xffffd9', '0xd4eeb3', '0x9ed9b8', '0xffffd9', '0xfeffd6', '0x83cebb', '0xe5f5b2', '0xffffd9', '0xffffd9', '0xf2fabc', '0xf1fabb', '0xb0e0b6', '0xd5efb3', '0x2073b2', '0xfafdcf', '0xffffd9', '0xf4fbc0', '0xedf8b1', '0xffffd9', '0xecf7b1', '0xf5fbc2', '0xfcfed1', '0xffffd9', '0xf9fdcb', '0xffffd9', '0xfeffd6', '0xa9ddb7', '0xffffd9', '0xdbf1b2', '0xffffd9', '0x40b5c4', '0xe5f5b2', '0x081d58', '0xf7fcc7', '0x2076b3', '0xf9fdcb', '0xfeffd6', '0xffffd9', '0x48b9c3', '0xceecb3', '0x7ecdbb', '0xf8fcca', '0xf5fbc2', '0xfcfed3']
colors = ['0xc8eac3', '0xf7fcf0', '0xd0ecc9', '0xf7fcf0', '0xccebc5', '0xf7fcf0', '0xe6f5e0', '0xedf8e7', '0xd7efd1', '0x9ed9b8', '0xf7fcf0', '0xdcf1d6', '0xcfecc8', '0x69c2ca', '0xccebc5', '0xe4f5df', '0xf5fbee', '0xcfecc8', '0xf7fcf0', '0xecf8e6', '0xccebc5', '0xf7fcf0', '0xc8eac3', '0xdaf1d5', '0xf7fcf0', '0xe6f5e0', '0xf7fcf0', '0xf7fcf0', '0xf7fcf0', '0xecf8e6', '0xf1f9ea', '0xebf7e5', '0xecf8e6', '0xeff9e9', '0xeaf7e4', '0x084889', '0xf6fbef', '0xf7fcf0', '0xd3eecc', '0xb7e3bc', '0xf7fcf0', '0xf6fbef', '0xaadeb6', '0xdcf1d6', '0xf7fcf0', '0xf7fcf0', '0xe6f6e1', '0xe6f5e0', '0xc0e6c0', '0xd3eecd', '0x3a9cc7', '0xf1faeb', '0xf7fcf0', '0xe9f6e3', '0xe0f3db', '0xf7fcf0', '0xdff3da', '0xeaf7e4', '0xf3faec', '0xf7fcf0', '0xeff9e9', '0xf7fcf0', '0xf6fbef', '0xbde5be', '0xf7fcf0', '0xd7efd1', '0xf7fcf0', '0x7accc4', '0xdcf1d6', '0x084081', '0xedf8e7', '0x3c9fc8', '0xeff9e9', '0xf6fbef', '0xf7fcf0', '0x80cec2', '0xd0ecc9', '0xa7ddb5', '0xeef9e8', '0xeaf7e4', '0xf3fbed']
# updated with chromophore
colors =['0xc8eac3', '0xf7fcf0', '0xd0ecc9', '0xf7fcf0', '0xccebc5', '0xf7fcf0', '0xe6f5e0', '0xedf8e7', '0xd7efd1', '0x9ed9b8', '0xf7fcf0', '0xdcf1d6', '0xcfecc8', '0x69c2ca', '0xccebc5', '0xe4f5df', '0xf5fbee', '0xcfecc8', '0xf7fcf0', '0xecf8e6', '0xccebc5', '0xf7fcf0', '0xc8eac3', '0xdaf1d5', '0xf7fcf0', '0xe6f5e0', '0xf7fcf0', '0xf7fcf0', '0xf7fcf0', '0xecf8e6', '0xf1f9ea', '0xebf7e5', '0xecf8e6', '0xeff9e9', '0xeaf7e4', '0xf7fcf0', '0x084688', '0xf7fcf0', '0xd3eecc', '0xb7e3bc', '0xf7fcf0', '0xf6fbef', '0xaadeb6', '0xdcf1d6', '0xf7fcf0', '0xf7fcf0', '0xe6f6e1', '0xe6f5e0', '0xc0e6c0', '0xd3eecd', '0x3a9cc7', '0xf1faeb', '0xf7fcf0', '0xe9f6e3', '0xe0f3db', '0xf7fcf0', '0xdff3da', '0xeaf7e4', '0xf3faec', '0xf7fcf0', '0xeff9e9', '0xf7fcf0', '0xf6fbef', '0xbde5be', '0xf7fcf0', '0xd7efd1', '0xf7fcf0', '0x7accc4', '0xdcf1d6', '0x084081', '0xedf8e7', '0x3c9fc8', '0xeff9e9', '0xf6fbef', '0xf7fcf0', '0x80cec2', '0xd0ecc9', '0xa7ddb5', '0xeef9e8', '0xeaf7e4', '0xf3fbed']
print(colors)
index = 30
for c in colors:
    cmd.do('color {0}, resi {1}'.format(c,index))
    index+=1
    time.sleep(.001)