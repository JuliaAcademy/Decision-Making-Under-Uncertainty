### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5771f1a3-a0df-4d65-9e7d-4ab45fa57360
begin
	using PlutoUI

	md"""
	# Particle Filtering
	##### Julia Academy: _Decision Making Under Uncertainty with POMDPs.jl_

	An introduction to _state estimation_ using [`ParticleFilters`](https://github.com/JuliaPOMDP/ParticleFilters.jl) and [`QuickPOMDPs`](https://github.com/JuliaPOMDP/QuickPOMDPs.jl).

	-- Robert Moss (Stanford University) as part of [_Julia Academy_](https://juliaacademy.com/) (Github: [mossr](https://github.com/mossr))
	"""
end

# ╔═╡ 7197cded-3ba1-4d0b-81b8-cd20130e3ad5
using POMDPs, QuickPOMDPs

# ╔═╡ afcad738-b323-4f31-a84b-ea8d4d9d3937
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# ╔═╡ 08f2254b-d7b6-46e2-8176-35e3c5561f97
using ParticleFilters

# ╔═╡ 19380e99-07e6-4490-a5e6-a451dde2e593
md"""
## Particle Filtering for State Estimation
To estimate the state of an agent defined by a POMDP, we can use a _particle filter_.$^1$

- _Particle filters_ represent the belief state as a collection of states.
- Each state in the approximated belief is called a _particle_.
- Useful in problems with large discrete states spaces or continuous problems not well approximated by linear-Gaussian dynamics.
"""

# ╔═╡ 96fe0aec-53e3-4aa9-971c-af6e4fa16d95
html"""
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="600" height="400" viewBox="0 0 2400 1600">
<defs>
  <clipPath id="clip050">
    <rect x="0" y="0" width="2400" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip050)" d="
M0 1600 L2400 1600 L2400 0 L0 0  Z
  " fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip051">
    <rect x="480" y="0" width="1681" height="1600"/>
  </clipPath>
</defs>
<path clip-path="url(#clip050)" d="
M542.811 1486.45 L1982.02 1486.45 L1982.02 47.2441 L542.811 47.2441  Z
  " fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<defs>
  <clipPath id="clip052">
    <rect x="542" y="47" width="1440" height="1440"/>
  </clipPath>
</defs>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,1486.45 542.811,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  782.679,1486.45 782.679,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  1022.55,1486.45 1022.55,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  1262.41,1486.45 1262.41,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  1502.28,1486.45 1502.28,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  1742.15,1486.45 1742.15,47.2441 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  1982.02,1486.45 1982.02,47.2441 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1486.45 1982.02,1486.45 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,47.2441 1982.02,47.2441 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1486.45 542.811,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  782.679,1486.45 782.679,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1022.55,1486.45 1022.55,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1262.41,1486.45 1262.41,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1502.28,1486.45 1502.28,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1742.15,1486.45 1742.15,1467.55 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1982.02,1486.45 1982.02,1467.55 
  "/>
<path clip-path="url(#clip050)" d="M533.943 1533.76 Q534.239 1534.05 534.239 1534.47 Q534.239 1534.89 533.943 1535.19 Q533.646 1535.48 533.226 1535.48 L501.557 1535.48 Q501.137 1535.48 500.84 1535.19 Q500.544 1534.89 500.544 1534.47 Q500.544 1534.05 500.84 1533.76 Q501.137 1533.46 501.557 1533.46 L533.226 1533.46 Q533.646 1533.46 533.943 1533.76 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M541.551 1518.24 L541.551 1516.66 Q547.628 1516.66 550.766 1513.42 Q551.63 1513.42 551.779 1513.62 Q551.927 1513.82 551.927 1514.73 L551.927 1543.12 Q551.927 1544.62 552.668 1545.09 Q553.409 1545.56 556.645 1545.56 L558.251 1545.56 L558.251 1547.12 Q556.472 1546.97 550.049 1546.97 Q543.627 1546.97 541.873 1547.12 L541.873 1545.56 L543.478 1545.56 Q546.665 1545.56 547.431 1545.12 Q548.197 1544.65 548.197 1543.12 L548.197 1516.91 Q545.553 1518.24 541.551 1518.24 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M564.871 1538.97 Q564.871 1537.46 565.687 1536.89 Q566.502 1536.3 567.366 1536.3 Q568.528 1536.3 569.194 1537.04 Q569.886 1537.76 569.886 1538.77 Q569.886 1539.78 569.194 1540.52 Q568.528 1541.24 567.366 1541.24 Q566.798 1541.24 566.502 1541.14 Q567.169 1543.46 569.194 1545.14 Q571.245 1546.82 573.938 1546.82 Q577.322 1546.82 579.348 1543.54 Q580.558 1541.41 580.558 1536.6 Q580.558 1532.35 579.644 1530.22 Q578.236 1526.99 575.346 1526.99 Q571.245 1526.99 568.824 1530.52 Q568.528 1530.96 568.182 1530.99 Q567.688 1530.99 567.564 1530.72 Q567.465 1530.42 567.465 1529.65 L567.465 1514.68 Q567.465 1513.47 567.959 1513.47 Q568.157 1513.47 568.577 1513.62 Q571.764 1515.03 575.296 1515.05 Q578.928 1515.05 582.188 1513.57 Q582.435 1513.42 582.584 1513.42 Q583.078 1513.42 583.102 1513.99 Q583.102 1514.19 582.683 1514.78 Q582.287 1515.35 581.423 1516.12 Q580.558 1516.86 579.446 1517.57 Q578.335 1518.29 576.704 1518.79 Q575.099 1519.25 573.32 1519.25 Q571.195 1519.25 569.022 1518.59 L569.022 1528.44 Q571.64 1525.87 575.444 1525.87 Q579.496 1525.87 582.287 1529.14 Q585.079 1532.4 585.079 1536.94 Q585.079 1541.71 581.769 1544.97 Q578.483 1548.23 574.036 1548.23 Q569.985 1548.23 567.416 1545.34 Q564.871 1542.45 564.871 1538.97 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M773.538 1533.76 Q773.835 1534.05 773.835 1534.47 Q773.835 1534.89 773.538 1535.19 Q773.242 1535.48 772.822 1535.48 L741.152 1535.48 Q740.732 1535.48 740.436 1535.19 Q740.139 1534.89 740.139 1534.47 Q740.139 1534.05 740.436 1533.76 Q740.732 1533.46 741.152 1533.46 L772.822 1533.46 Q773.242 1533.46 773.538 1533.76 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M781.147 1518.24 L781.147 1516.66 Q787.224 1516.66 790.361 1513.42 Q791.226 1513.42 791.374 1513.62 Q791.522 1513.82 791.522 1514.73 L791.522 1543.12 Q791.522 1544.62 792.264 1545.09 Q793.005 1545.56 796.241 1545.56 L797.846 1545.56 L797.846 1547.12 Q796.068 1546.97 789.645 1546.97 Q783.222 1546.97 781.468 1547.12 L781.468 1545.56 L783.074 1545.56 Q786.261 1545.56 787.026 1545.12 Q787.792 1544.65 787.792 1543.12 L787.792 1516.91 Q785.149 1518.24 781.147 1518.24 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M803.923 1530.94 Q803.923 1523.5 805.801 1519.45 Q808.42 1513.42 814.595 1513.42 Q815.905 1513.42 817.263 1513.8 Q818.647 1514.14 820.376 1515.5 Q822.13 1516.86 823.192 1519.08 Q825.218 1523.38 825.218 1530.94 Q825.218 1538.33 823.34 1542.35 Q820.598 1548.23 814.546 1548.23 Q812.273 1548.23 809.951 1547.07 Q807.654 1545.91 806.196 1543.12 Q803.923 1538.94 803.923 1530.94 M808.123 1530.32 Q808.123 1537.95 808.667 1540.99 Q809.284 1544.28 810.964 1545.71 Q812.668 1547.12 814.546 1547.12 Q816.572 1547.12 818.251 1545.61 Q819.956 1544.08 820.475 1540.8 Q821.043 1537.56 821.018 1530.32 Q821.018 1523.28 820.524 1520.46 Q819.857 1517.18 818.079 1515.87 Q816.325 1514.54 814.546 1514.54 Q813.879 1514.54 813.163 1514.73 Q812.471 1514.93 811.458 1515.5 Q810.445 1516.07 809.655 1517.48 Q808.889 1518.88 808.518 1521.01 Q808.123 1523.75 808.123 1530.32 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1026.33 1533.76 Q1026.62 1534.05 1026.62 1534.47 Q1026.62 1534.89 1026.33 1535.19 Q1026.03 1535.48 1025.61 1535.48 L993.939 1535.48 Q993.519 1535.48 993.223 1535.19 Q992.927 1534.89 992.927 1534.47 Q992.927 1534.05 993.223 1533.76 Q993.519 1533.46 993.939 1533.46 L1025.61 1533.46 Q1026.03 1533.46 1026.33 1533.76 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1031.96 1538.97 Q1031.96 1537.46 1032.77 1536.89 Q1033.59 1536.3 1034.45 1536.3 Q1035.61 1536.3 1036.28 1537.04 Q1036.97 1537.76 1036.97 1538.77 Q1036.97 1539.78 1036.28 1540.52 Q1035.61 1541.24 1034.45 1541.24 Q1033.88 1541.24 1033.59 1541.14 Q1034.26 1543.46 1036.28 1545.14 Q1038.33 1546.82 1041.02 1546.82 Q1044.41 1546.82 1046.43 1543.54 Q1047.64 1541.41 1047.64 1536.6 Q1047.64 1532.35 1046.73 1530.22 Q1045.32 1526.99 1042.43 1526.99 Q1038.33 1526.99 1035.91 1530.52 Q1035.61 1530.96 1035.27 1530.99 Q1034.77 1530.99 1034.65 1530.72 Q1034.55 1530.42 1034.55 1529.65 L1034.55 1514.68 Q1034.55 1513.47 1035.05 1513.47 Q1035.24 1513.47 1035.66 1513.62 Q1038.85 1515.03 1042.38 1515.05 Q1046.01 1515.05 1049.27 1513.57 Q1049.52 1513.42 1049.67 1513.42 Q1050.16 1513.42 1050.19 1513.99 Q1050.19 1514.19 1049.77 1514.78 Q1049.37 1515.35 1048.51 1516.12 Q1047.64 1516.86 1046.53 1517.57 Q1045.42 1518.29 1043.79 1518.79 Q1042.19 1519.25 1040.41 1519.25 Q1038.28 1519.25 1036.11 1518.59 L1036.11 1528.44 Q1038.73 1525.87 1042.53 1525.87 Q1046.58 1525.87 1049.37 1529.14 Q1052.17 1532.4 1052.17 1536.94 Q1052.17 1541.71 1048.85 1544.97 Q1045.57 1548.23 1041.12 1548.23 Q1037.07 1548.23 1034.5 1545.34 Q1031.96 1542.45 1031.96 1538.97 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1251.77 1530.94 Q1251.77 1523.5 1253.64 1519.45 Q1256.26 1513.42 1262.44 1513.42 Q1263.75 1513.42 1265.11 1513.8 Q1266.49 1514.14 1268.22 1515.5 Q1269.97 1516.86 1271.03 1519.08 Q1273.06 1523.38 1273.06 1530.94 Q1273.06 1538.33 1271.18 1542.35 Q1268.44 1548.23 1262.39 1548.23 Q1260.12 1548.23 1257.79 1547.07 Q1255.5 1545.91 1254.04 1543.12 Q1251.77 1538.94 1251.77 1530.94 M1255.97 1530.32 Q1255.97 1537.95 1256.51 1540.99 Q1257.13 1544.28 1258.81 1545.71 Q1260.51 1547.12 1262.39 1547.12 Q1264.41 1547.12 1266.09 1545.61 Q1267.8 1544.08 1268.32 1540.8 Q1268.89 1537.56 1268.86 1530.32 Q1268.86 1523.28 1268.37 1520.46 Q1267.7 1517.18 1265.92 1515.87 Q1264.17 1514.54 1262.39 1514.54 Q1261.72 1514.54 1261.01 1514.73 Q1260.31 1514.93 1259.3 1515.5 Q1258.29 1516.07 1257.5 1517.48 Q1256.73 1518.88 1256.36 1521.01 Q1255.97 1523.75 1255.97 1530.32 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1492.18 1538.97 Q1492.18 1537.46 1492.99 1536.89 Q1493.81 1536.3 1494.67 1536.3 Q1495.83 1536.3 1496.5 1537.04 Q1497.19 1537.76 1497.19 1538.77 Q1497.19 1539.78 1496.5 1540.52 Q1495.83 1541.24 1494.67 1541.24 Q1494.1 1541.24 1493.81 1541.14 Q1494.47 1543.46 1496.5 1545.14 Q1498.55 1546.82 1501.24 1546.82 Q1504.63 1546.82 1506.65 1543.54 Q1507.86 1541.41 1507.86 1536.6 Q1507.86 1532.35 1506.95 1530.22 Q1505.54 1526.99 1502.65 1526.99 Q1498.55 1526.99 1496.13 1530.52 Q1495.83 1530.96 1495.49 1530.99 Q1494.99 1530.99 1494.87 1530.72 Q1494.77 1530.42 1494.77 1529.65 L1494.77 1514.68 Q1494.77 1513.47 1495.26 1513.47 Q1495.46 1513.47 1495.88 1513.62 Q1499.07 1515.03 1502.6 1515.05 Q1506.23 1515.05 1509.49 1513.57 Q1509.74 1513.42 1509.89 1513.42 Q1510.38 1513.42 1510.41 1513.99 Q1510.41 1514.19 1509.99 1514.78 Q1509.59 1515.35 1508.73 1516.12 Q1507.86 1516.86 1506.75 1517.57 Q1505.64 1518.29 1504.01 1518.79 Q1502.4 1519.25 1500.63 1519.25 Q1498.5 1519.25 1496.33 1518.59 L1496.33 1528.44 Q1498.95 1525.87 1502.75 1525.87 Q1506.8 1525.87 1509.59 1529.14 Q1512.38 1532.4 1512.38 1536.94 Q1512.38 1541.71 1509.07 1544.97 Q1505.79 1548.23 1501.34 1548.23 Q1497.29 1548.23 1494.72 1545.34 Q1492.18 1542.45 1492.18 1538.97 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1720.11 1518.24 L1720.11 1516.66 Q1726.19 1516.66 1729.33 1513.42 Q1730.19 1513.42 1730.34 1513.62 Q1730.49 1513.82 1730.49 1514.73 L1730.49 1543.12 Q1730.49 1544.62 1731.23 1545.09 Q1731.97 1545.56 1735.21 1545.56 L1736.81 1545.56 L1736.81 1547.12 Q1735.03 1546.97 1728.61 1546.97 Q1722.19 1546.97 1720.43 1547.12 L1720.43 1545.56 L1722.04 1545.56 Q1725.23 1545.56 1725.99 1545.12 Q1726.76 1544.65 1726.76 1543.12 L1726.76 1516.91 Q1724.11 1518.24 1720.11 1518.24 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1742.89 1530.94 Q1742.89 1523.5 1744.77 1519.45 Q1747.38 1513.42 1753.56 1513.42 Q1754.87 1513.42 1756.23 1513.8 Q1757.61 1514.14 1759.34 1515.5 Q1761.1 1516.86 1762.16 1519.08 Q1764.18 1523.38 1764.18 1530.94 Q1764.18 1538.33 1762.31 1542.35 Q1759.56 1548.23 1753.51 1548.23 Q1751.24 1548.23 1748.92 1547.07 Q1746.62 1545.91 1745.16 1543.12 Q1742.89 1538.94 1742.89 1530.94 M1747.09 1530.32 Q1747.09 1537.95 1747.63 1540.99 Q1748.25 1544.28 1749.93 1545.71 Q1751.63 1547.12 1753.51 1547.12 Q1755.54 1547.12 1757.22 1545.61 Q1758.92 1544.08 1759.44 1540.8 Q1760.01 1537.56 1759.98 1530.32 Q1759.98 1523.28 1759.49 1520.46 Q1758.82 1517.18 1757.04 1515.87 Q1755.29 1514.54 1753.51 1514.54 Q1752.84 1514.54 1752.13 1514.73 Q1751.44 1514.93 1750.42 1515.5 Q1749.41 1516.07 1748.62 1517.48 Q1747.85 1518.88 1747.48 1521.01 Q1747.09 1523.75 1747.09 1530.32 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1960.25 1518.24 L1960.25 1516.66 Q1966.33 1516.66 1969.47 1513.42 Q1970.33 1513.42 1970.48 1513.62 Q1970.63 1513.82 1970.63 1514.73 L1970.63 1543.12 Q1970.63 1544.62 1971.37 1545.09 Q1972.11 1545.56 1975.35 1545.56 L1976.95 1545.56 L1976.95 1547.12 Q1975.17 1546.97 1968.75 1546.97 Q1962.33 1546.97 1960.57 1547.12 L1960.57 1545.56 L1962.18 1545.56 Q1965.37 1545.56 1966.13 1545.12 Q1966.9 1544.65 1966.9 1543.12 L1966.9 1516.91 Q1964.25 1518.24 1960.25 1518.24 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1983.57 1538.97 Q1983.57 1537.46 1984.39 1536.89 Q1985.2 1536.3 1986.07 1536.3 Q1987.23 1536.3 1987.89 1537.04 Q1988.59 1537.76 1988.59 1538.77 Q1988.59 1539.78 1987.89 1540.52 Q1987.23 1541.24 1986.07 1541.24 Q1985.5 1541.24 1985.2 1541.14 Q1985.87 1543.46 1987.89 1545.14 Q1989.94 1546.82 1992.64 1546.82 Q1996.02 1546.82 1998.05 1543.54 Q1999.26 1541.41 1999.26 1536.6 Q1999.26 1532.35 1998.34 1530.22 Q1996.94 1526.99 1994.05 1526.99 Q1989.94 1526.99 1987.52 1530.52 Q1987.23 1530.96 1986.88 1530.99 Q1986.39 1530.99 1986.26 1530.72 Q1986.17 1530.42 1986.17 1529.65 L1986.17 1514.68 Q1986.17 1513.47 1986.66 1513.47 Q1986.86 1513.47 1987.28 1513.62 Q1990.46 1515.03 1994 1515.05 Q1997.63 1515.05 2000.89 1513.57 Q2001.14 1513.42 2001.28 1513.42 Q2001.78 1513.42 2001.8 1513.99 Q2001.8 1514.19 2001.38 1514.78 Q2000.99 1515.35 2000.12 1516.12 Q1999.26 1516.86 1998.15 1517.57 Q1997.03 1518.29 1995.4 1518.79 Q1993.8 1519.25 1992.02 1519.25 Q1989.9 1519.25 1987.72 1518.59 L1987.72 1528.44 Q1990.34 1525.87 1994.14 1525.87 Q1998.2 1525.87 2000.99 1529.14 Q2003.78 1532.4 2003.78 1536.94 Q2003.78 1541.71 2000.47 1544.97 Q1997.18 1548.23 1992.74 1548.23 Q1988.68 1548.23 1986.12 1545.34 Q1983.57 1542.45 1983.57 1538.97 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,1486.45 1982.02,1486.45 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,1246.58 1982.02,1246.58 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,1006.71 1982.02,1006.71 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,766.846 1982.02,766.846 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,526.979 1982.02,526.979 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,287.111 1982.02,287.111 
  "/>
<polyline clip-path="url(#clip052)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:2; stroke-opacity:0.1; fill:none" points="
  542.811,47.2441 1982.02,47.2441 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1486.45 542.811,47.2441 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1982.02,1486.45 1982.02,47.2441 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1486.45 555.283,1486.45 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1246.58 555.283,1246.58 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,1006.71 555.283,1006.71 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,766.846 555.283,766.846 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,526.979 555.283,526.979 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,287.111 555.283,287.111 
  "/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  542.811,47.2441 555.283,47.2441 
  "/>
<path clip-path="url(#clip050)" d="M455.675 1490.36 Q455.972 1490.66 455.972 1491.08 Q455.972 1491.5 455.675 1491.8 Q455.379 1492.09 454.959 1492.09 L423.289 1492.09 Q422.869 1492.09 422.573 1491.8 Q422.276 1491.5 422.276 1491.08 Q422.276 1490.66 422.573 1490.36 Q422.869 1490.07 423.289 1490.07 L454.959 1490.07 Q455.379 1490.07 455.675 1490.36 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M463.284 1474.85 L463.284 1473.27 Q469.361 1473.27 472.498 1470.03 Q473.363 1470.03 473.511 1470.23 Q473.659 1470.43 473.659 1471.34 L473.659 1499.73 Q473.659 1501.23 474.401 1501.7 Q475.142 1502.17 478.378 1502.17 L479.983 1502.17 L479.983 1503.73 Q478.205 1503.58 471.782 1503.58 Q465.359 1503.58 463.605 1503.73 L463.605 1502.17 L465.211 1502.17 Q468.398 1502.17 469.163 1501.73 Q469.929 1501.26 469.929 1499.73 L469.929 1473.52 Q467.286 1474.85 463.284 1474.85 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M486.604 1495.58 Q486.604 1494.07 487.419 1493.5 Q488.234 1492.91 489.099 1492.91 Q490.26 1492.91 490.927 1493.65 Q491.619 1494.37 491.619 1495.38 Q491.619 1496.39 490.927 1497.13 Q490.26 1497.85 489.099 1497.85 Q488.531 1497.85 488.234 1497.75 Q488.901 1500.07 490.927 1501.75 Q492.977 1503.43 495.67 1503.43 Q499.054 1503.43 501.08 1500.15 Q502.291 1498.02 502.291 1493.2 Q502.291 1488.96 501.377 1486.83 Q499.968 1483.59 497.078 1483.59 Q492.977 1483.59 490.557 1487.13 Q490.26 1487.57 489.914 1487.6 Q489.42 1487.6 489.297 1487.32 Q489.198 1487.03 489.198 1486.26 L489.198 1471.29 Q489.198 1470.08 489.692 1470.08 Q489.89 1470.08 490.309 1470.23 Q493.496 1471.64 497.029 1471.66 Q500.66 1471.66 503.921 1470.18 Q504.168 1470.03 504.316 1470.03 Q504.81 1470.03 504.835 1470.6 Q504.835 1470.8 504.415 1471.39 Q504.02 1471.96 503.155 1472.73 Q502.291 1473.47 501.179 1474.18 Q500.067 1474.9 498.437 1475.39 Q496.831 1475.86 495.053 1475.86 Q492.928 1475.86 490.754 1475.2 L490.754 1485.05 Q493.373 1482.48 497.177 1482.48 Q501.228 1482.48 504.02 1485.74 Q506.811 1489 506.811 1493.55 Q506.811 1498.32 503.501 1501.58 Q500.216 1504.84 495.769 1504.84 Q491.718 1504.84 489.148 1501.95 Q486.604 1499.06 486.604 1495.58 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M455.132 1250.5 Q455.428 1250.79 455.428 1251.21 Q455.428 1251.63 455.132 1251.93 Q454.835 1252.23 454.415 1252.23 L422.746 1252.23 Q422.326 1252.23 422.029 1251.93 Q421.733 1251.63 421.733 1251.21 Q421.733 1250.79 422.029 1250.5 Q422.326 1250.2 422.746 1250.2 L454.415 1250.2 Q454.835 1250.2 455.132 1250.5 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M462.741 1234.98 L462.741 1233.4 Q468.818 1233.4 471.955 1230.17 Q472.819 1230.17 472.968 1230.36 Q473.116 1230.56 473.116 1231.47 L473.116 1259.86 Q473.116 1261.37 473.857 1261.83 Q474.598 1262.3 477.834 1262.3 L479.44 1262.3 L479.44 1263.86 Q477.661 1263.71 471.238 1263.71 Q464.816 1263.71 463.062 1263.86 L463.062 1262.3 L464.667 1262.3 Q467.854 1262.3 468.62 1261.86 Q469.386 1261.39 469.386 1259.86 L469.386 1233.65 Q466.742 1234.98 462.741 1234.98 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M485.517 1247.68 Q485.517 1240.24 487.394 1236.19 Q490.013 1230.17 496.189 1230.17 Q497.498 1230.17 498.857 1230.54 Q500.24 1230.88 501.969 1232.24 Q503.723 1233.6 504.786 1235.82 Q506.811 1240.12 506.811 1247.68 Q506.811 1255.07 504.934 1259.09 Q502.192 1264.97 496.139 1264.97 Q493.867 1264.97 491.545 1263.81 Q489.247 1262.65 487.79 1259.86 Q485.517 1255.68 485.517 1247.68 M489.717 1247.06 Q489.717 1254.7 490.26 1257.73 Q490.878 1261.02 492.557 1262.45 Q494.262 1263.86 496.139 1263.86 Q498.165 1263.86 499.845 1262.35 Q501.549 1260.82 502.068 1257.54 Q502.636 1254.3 502.612 1247.06 Q502.612 1240.02 502.118 1237.21 Q501.451 1233.92 499.672 1232.61 Q497.918 1231.28 496.139 1231.28 Q495.472 1231.28 494.756 1231.47 Q494.064 1231.67 493.052 1232.24 Q492.039 1232.81 491.248 1234.22 Q490.482 1235.62 490.112 1237.75 Q489.717 1240.49 489.717 1247.06 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M480.972 1010.63 Q481.268 1010.93 481.268 1011.35 Q481.268 1011.77 480.972 1012.06 Q480.675 1012.36 480.255 1012.36 L448.585 1012.36 Q448.166 1012.36 447.869 1012.06 Q447.573 1011.77 447.573 1011.35 Q447.573 1010.93 447.869 1010.63 Q448.166 1010.33 448.585 1010.33 L480.255 1010.33 Q480.675 1010.33 480.972 1010.63 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M486.604 1015.84 Q486.604 1014.33 487.419 1013.77 Q488.234 1013.17 489.099 1013.17 Q490.26 1013.17 490.927 1013.91 Q491.619 1014.63 491.619 1015.64 Q491.619 1016.66 490.927 1017.4 Q490.26 1018.11 489.099 1018.11 Q488.531 1018.11 488.234 1018.02 Q488.901 1020.34 490.927 1022.02 Q492.977 1023.7 495.67 1023.7 Q499.054 1023.7 501.08 1020.41 Q502.291 1018.29 502.291 1013.47 Q502.291 1009.22 501.377 1007.1 Q499.968 1003.86 497.078 1003.86 Q492.977 1003.86 490.557 1007.39 Q490.26 1007.84 489.914 1007.86 Q489.42 1007.86 489.297 1007.59 Q489.198 1007.29 489.198 1006.53 L489.198 991.558 Q489.198 990.347 489.692 990.347 Q489.89 990.347 490.309 990.496 Q493.496 991.904 497.029 991.928 Q500.66 991.928 503.921 990.446 Q504.168 990.298 504.316 990.298 Q504.81 990.298 504.835 990.866 Q504.835 991.064 504.415 991.657 Q504.02 992.225 503.155 992.991 Q502.291 993.732 501.179 994.448 Q500.067 995.164 498.437 995.659 Q496.831 996.128 495.053 996.128 Q492.928 996.128 490.754 995.461 L490.754 1005.32 Q493.373 1002.75 497.177 1002.75 Q501.228 1002.75 504.02 1006.01 Q506.811 1009.27 506.811 1013.82 Q506.811 1018.58 503.501 1021.84 Q500.216 1025.1 495.769 1025.1 Q491.718 1025.1 489.148 1022.21 Q486.604 1019.32 486.604 1015.84 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M485.517 767.945 Q485.517 760.51 487.394 756.458 Q490.013 750.431 496.189 750.431 Q497.498 750.431 498.857 750.801 Q500.24 751.147 501.969 752.506 Q503.723 753.864 504.786 756.088 Q506.811 760.386 506.811 767.945 Q506.811 775.332 504.934 779.358 Q502.192 785.238 496.139 785.238 Q493.867 785.238 491.545 784.077 Q489.247 782.916 487.79 780.124 Q485.517 775.949 485.517 767.945 M489.717 767.328 Q489.717 774.961 490.26 778 Q490.878 781.285 492.557 782.718 Q494.262 784.126 496.139 784.126 Q498.165 784.126 499.845 782.619 Q501.549 781.087 502.068 777.802 Q502.636 774.566 502.612 767.328 Q502.612 760.287 502.118 757.471 Q501.451 754.186 499.672 752.876 Q497.918 751.542 496.139 751.542 Q495.472 751.542 494.756 751.74 Q494.064 751.938 493.052 752.506 Q492.039 753.074 491.248 754.482 Q490.482 755.89 490.112 758.015 Q489.717 760.757 489.717 767.328 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M486.604 536.107 Q486.604 534.6 487.419 534.031 Q488.234 533.439 489.099 533.439 Q490.26 533.439 490.927 534.18 Q491.619 534.896 491.619 535.909 Q491.619 536.922 490.927 537.663 Q490.26 538.379 489.099 538.379 Q488.531 538.379 488.234 538.28 Q488.901 540.603 490.927 542.282 Q492.977 543.962 495.67 543.962 Q499.054 543.962 501.08 540.677 Q502.291 538.552 502.291 533.735 Q502.291 529.486 501.377 527.362 Q499.968 524.125 497.078 524.125 Q492.977 524.125 490.557 527.658 Q490.26 528.103 489.914 528.127 Q489.42 528.127 489.297 527.856 Q489.198 527.559 489.198 526.793 L489.198 511.823 Q489.198 510.613 489.692 510.613 Q489.89 510.613 490.309 510.761 Q493.496 512.169 497.029 512.194 Q500.66 512.194 503.921 510.712 Q504.168 510.563 504.316 510.563 Q504.81 510.563 504.835 511.131 Q504.835 511.329 504.415 511.922 Q504.02 512.49 503.155 513.256 Q502.291 513.997 501.179 514.713 Q500.067 515.43 498.437 515.924 Q496.831 516.393 495.053 516.393 Q492.928 516.393 490.754 515.726 L490.754 525.583 Q493.373 523.014 497.177 523.014 Q501.228 523.014 504.02 526.275 Q506.811 529.535 506.811 534.081 Q506.811 538.849 503.501 542.109 Q500.216 545.37 495.769 545.37 Q491.718 545.37 489.148 542.48 Q486.604 539.59 486.604 536.107 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M462.741 275.513 L462.741 273.932 Q468.818 273.932 471.955 270.696 Q472.819 270.696 472.968 270.894 Q473.116 271.091 473.116 272.005 L473.116 300.389 Q473.116 301.896 473.857 302.366 Q474.598 302.835 477.834 302.835 L479.44 302.835 L479.44 304.391 Q477.661 304.243 471.238 304.243 Q464.816 304.243 463.062 304.391 L463.062 302.835 L464.667 302.835 Q467.854 302.835 468.62 302.39 Q469.386 301.921 469.386 300.389 L469.386 274.179 Q466.742 275.513 462.741 275.513 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M485.517 288.211 Q485.517 280.775 487.394 276.724 Q490.013 270.696 496.189 270.696 Q497.498 270.696 498.857 271.067 Q500.24 271.412 501.969 272.771 Q503.723 274.13 504.786 276.353 Q506.811 280.651 506.811 288.211 Q506.811 295.597 504.934 299.624 Q502.192 305.503 496.139 305.503 Q493.867 305.503 491.545 304.342 Q489.247 303.181 487.79 300.389 Q485.517 296.215 485.517 288.211 M489.717 287.593 Q489.717 295.226 490.26 298.265 Q490.878 301.551 492.557 302.983 Q494.262 304.391 496.139 304.391 Q498.165 304.391 499.845 302.884 Q501.549 301.353 502.068 298.067 Q502.636 294.831 502.612 287.593 Q502.612 280.553 502.118 277.736 Q501.451 274.451 499.672 273.142 Q497.918 271.808 496.139 271.808 Q495.472 271.808 494.756 272.005 Q494.064 272.203 493.052 272.771 Q492.039 273.339 491.248 274.747 Q490.482 276.155 490.112 278.28 Q489.717 281.022 489.717 287.593 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M463.284 35.6459 L463.284 34.0649 Q469.361 34.0649 472.498 30.8287 Q473.363 30.8287 473.511 31.0263 Q473.659 31.224 473.659 32.138 L473.659 60.5222 Q473.659 62.0291 474.401 62.4984 Q475.142 62.9678 478.378 62.9678 L479.983 62.9678 L479.983 64.5241 Q478.205 64.3759 471.782 64.3759 Q465.359 64.3759 463.605 64.5241 L463.605 62.9678 L465.211 62.9678 Q468.398 62.9678 469.163 62.5231 Q469.929 62.0538 469.929 60.5222 L469.929 34.3119 Q467.286 35.6459 463.284 35.6459 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M486.604 56.372 Q486.604 54.8651 487.419 54.2969 Q488.234 53.704 489.099 53.704 Q490.26 53.704 490.927 54.4451 Q491.619 55.1615 491.619 56.1744 Q491.619 57.1872 490.927 57.9283 Q490.26 58.6447 489.099 58.6447 Q488.531 58.6447 488.234 58.5459 Q488.901 60.868 490.927 62.5478 Q492.977 64.2277 495.67 64.2277 Q499.054 64.2277 501.08 60.9421 Q502.291 58.8176 502.291 54.0005 Q502.291 49.7515 501.377 47.627 Q499.968 44.3909 497.078 44.3909 Q492.977 44.3909 490.557 47.9234 Q490.26 48.3681 489.914 48.3928 Q489.42 48.3928 489.297 48.1211 Q489.198 47.8246 489.198 47.0588 L489.198 32.0886 Q489.198 30.8781 489.692 30.8781 Q489.89 30.8781 490.309 31.0263 Q493.496 32.4344 497.029 32.4591 Q500.66 32.4591 503.921 30.9769 Q504.168 30.8287 504.316 30.8287 Q504.81 30.8287 504.835 31.3969 Q504.835 31.5945 504.415 32.1874 Q504.02 32.7556 503.155 33.5214 Q502.291 34.2625 501.179 34.9789 Q500.067 35.6953 498.437 36.1893 Q496.831 36.6587 495.053 36.6587 Q492.928 36.6587 490.754 35.9917 L490.754 45.8484 Q493.373 43.2792 497.177 43.2792 Q501.228 43.2792 504.02 46.54 Q506.811 49.8009 506.811 54.3463 Q506.811 59.1141 503.501 62.3749 Q500.216 65.6357 495.769 65.6357 Q491.718 65.6357 489.148 62.7455 Q486.604 59.8552 486.604 56.372 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><circle clip-path="url(#clip052)" cx="1316.75" cy="839.194" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.75" cy="839.194" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.29" cy="787.531" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.29" cy="787.531" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.29" cy="787.531" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.1" cy="815.846" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.1" cy="815.846" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.1" cy="815.846" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1399.5" cy="807.923" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1399.5" cy="807.923" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1418.49" cy="828.196" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.83" cy="855.061" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.9" cy="822.654" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.9" cy="822.654" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1415.6" cy="848.93" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1453.73" cy="759.82" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1455.29" cy="803.306" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1421.69" cy="795.536" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1416.43" cy="728.067" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.08" cy="803.726" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.08" cy="803.726" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.08" cy="803.726" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1301.02" cy="870.768" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.58" cy="849.432" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.72" cy="754.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1302.36" cy="804.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1302.36" cy="804.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1302.36" cy="804.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1422.24" cy="853.345" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.18" cy="776.808" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.18" cy="776.808" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.18" cy="776.808" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.18" cy="776.808" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.11" cy="882.507" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1291.44" cy="806.392" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1291.44" cy="806.392" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.04" cy="752.881" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1387.21" cy="847.647" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.47" cy="850.234" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.07" cy="812.38" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.07" cy="812.38" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.07" cy="812.38" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.09" cy="741.66" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.09" cy="741.66" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.09" cy="741.66" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.09" cy="741.66" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.09" cy="741.66" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1407.73" cy="804.505" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1257.7" cy="744.171" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.69" cy="767.268" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.69" cy="767.268" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.69" cy="767.268" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.69" cy="767.268" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1297.64" cy="826.279" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.32" cy="808.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.32" cy="808.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.32" cy="808.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.02" cy="807.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.02" cy="807.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.02" cy="807.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.02" cy="807.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.39" cy="798.926" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.39" cy="798.926" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.39" cy="798.926" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.39" cy="798.926" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.47" cy="835.761" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1273.99" cy="799.281" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1273.99" cy="799.281" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.17" cy="833.616" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1303.49" cy="783.029" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1303.49" cy="783.029" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1303.49" cy="783.029" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1303.49" cy="783.029" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.86" cy="805.481" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.86" cy="805.481" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.86" cy="805.481" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.86" cy="805.481" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.23" cy="841.504" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.46" cy="773.593" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.46" cy="773.593" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.46" cy="773.593" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.46" cy="773.593" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.46" cy="773.593" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1314.95" cy="824.196" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1314.95" cy="824.196" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1314.95" cy="824.196" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1269.77" cy="779.237" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1269.77" cy="779.237" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.66" cy="853.349" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.2" cy="821.766" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.2" cy="821.766" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.71" cy="841.17" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.93" cy="847.631" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.72" cy="814.124" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.72" cy="814.124" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.47" cy="854.684" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.91" cy="776.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.91" cy="776.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.91" cy="776.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.91" cy="776.814" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.92" cy="776.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.92" cy="776.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.92" cy="776.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.92" cy="776.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.92" cy="776.414" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.17" cy="785.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.17" cy="785.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.17" cy="785.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.17" cy="785.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1248.63" cy="881.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.73" cy="808.389" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.73" cy="808.389" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1305.33" cy="824.278" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.16" cy="820.498" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.16" cy="820.498" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1437.49" cy="815.219" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1251.85" cy="788.087" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370.62" cy="857.878" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.71" cy="713.813" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1293.71" cy="713.813" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.63" cy="678.509" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.76" cy="864.805" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1243.99" cy="780.997" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.16" cy="778.187" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.16" cy="778.187" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.16" cy="778.187" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.16" cy="778.187" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.16" cy="778.187" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.28" cy="803.79" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.28" cy="803.79" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.28" cy="803.79" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.28" cy="803.79" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.95" cy="814.215" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.95" cy="814.215" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.95" cy="814.215" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.24" cy="825.371" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.24" cy="825.371" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.34" cy="762.003" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.29" cy="787.968" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.29" cy="787.968" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.29" cy="787.968" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.29" cy="787.968" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.05" cy="793.468" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.05" cy="793.468" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.05" cy="793.468" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.05" cy="793.468" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.13" cy="760.58" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.13" cy="760.58" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.13" cy="760.58" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.13" cy="760.58" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.66" cy="795.724" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.66" cy="795.724" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.66" cy="795.724" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.66" cy="795.724" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.55" cy="814.159" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.55" cy="814.159" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.55" cy="814.159" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.17" cy="808.402" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.17" cy="808.402" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.17" cy="808.402" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1294.23" cy="858.036" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1440.03" cy="799.807" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.06" cy="815.864" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.06" cy="815.864" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.8" cy="781.442" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.8" cy="781.442" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.87" cy="795.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.87" cy="795.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.87" cy="795.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.87" cy="795.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.51" cy="813.615" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.51" cy="813.615" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.51" cy="813.615" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.51" cy="813.615" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.14" cy="867.131" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.71" cy="839.775" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.95" cy="829.815" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.95" cy="829.815" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.19" cy="833.006" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.19" cy="833.006" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1383.31" cy="823.443" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1383.31" cy="823.443" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1417.11" cy="777.495" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1417.11" cy="777.495" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1418.08" cy="771.044" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.8" cy="746.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.8" cy="746.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.8" cy="746.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.8" cy="746.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.8" cy="746.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.84" cy="843.57" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.9" cy="826.817" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.9" cy="826.817" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.9" cy="826.817" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.08" cy="795.158" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.08" cy="795.158" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1385.44" cy="784.629" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1385.44" cy="784.629" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1385.44" cy="784.629" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1283.28" cy="755.565" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1283.28" cy="755.565" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1283.28" cy="755.565" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1429.56" cy="836.984" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.28" cy="805.142" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.28" cy="805.142" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370" cy="773.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370" cy="773.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370" cy="773.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370" cy="773.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370" cy="773.888" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.25" cy="860.108" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.13" cy="785.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.13" cy="785.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.13" cy="785.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.78" cy="862.137" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1373.07" cy="776.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1373.07" cy="776.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1373.07" cy="776.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1373.07" cy="776.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1373.07" cy="776.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1344.24" cy="820.054" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1344.24" cy="820.054" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.65" cy="848.053" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.57" cy="833.012" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1352.63" cy="832.083" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1352.63" cy="832.083" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.49" cy="825.26" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.49" cy="825.26" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.35" cy="855.625" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.91" cy="814.241" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.91" cy="814.241" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.91" cy="814.241" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.68" cy="808.064" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.68" cy="808.064" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.68" cy="808.064" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1311.11" cy="881.634" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.65" cy="810.789" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.65" cy="810.789" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.65" cy="810.789" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.45" cy="745.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.45" cy="745.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.45" cy="745.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.45" cy="745.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.45" cy="745.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1405.48" cy="841.8" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.95" cy="779.561" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.95" cy="779.561" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.95" cy="779.561" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.98" cy="824.694" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.98" cy="824.694" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1429.85" cy="788.249" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.89" cy="866.777" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1263.45" cy="809.057" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.7" cy="868.049" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.46" cy="789.171" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.46" cy="789.171" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.46" cy="789.171" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.46" cy="789.171" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.26" cy="833.377" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.26" cy="833.377" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1401.93" cy="815.672" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.29" cy="780.213" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.29" cy="780.213" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.29" cy="780.213" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.98" cy="889.489" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.88" cy="801.113" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.88" cy="801.113" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.88" cy="801.113" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1313.5" cy="835.627" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1289.79" cy="858.412" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.61" cy="841.04" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.7" cy="783.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.7" cy="783.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.7" cy="783.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.7" cy="783.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.7" cy="783.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.53" cy="809.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.53" cy="809.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.72" cy="761.746" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.72" cy="761.746" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.72" cy="761.746" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.72" cy="761.746" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1404.23" cy="824.493" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.93" cy="845.505" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1363.19" cy="833.665" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1363.19" cy="833.665" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.33" cy="811.772" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.33" cy="811.772" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.33" cy="811.772" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.46" cy="764.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.44" cy="811.977" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.44" cy="811.977" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.44" cy="811.977" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.78" cy="832.612" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.8" cy="735.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1309.43" cy="824.201" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1311.5" cy="807.734" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1311.5" cy="807.734" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1311.5" cy="807.734" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.88" cy="855.968" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.57" cy="755.725" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.57" cy="755.725" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.57" cy="755.725" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.57" cy="755.725" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.57" cy="755.725" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.18" cy="791.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.18" cy="791.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.18" cy="791.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.18" cy="791.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.18" cy="791.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.15" cy="839.392" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.38" cy="774.931" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.38" cy="774.931" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.38" cy="774.931" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.38" cy="774.931" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.1" cy="810.617" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.1" cy="810.617" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.53" cy="820.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.53" cy="820.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.53" cy="820.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.4" cy="805.225" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.4" cy="805.225" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.4" cy="805.225" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.43" cy="771.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.43" cy="771.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.43" cy="771.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.43" cy="771.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.44" cy="814.553" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.44" cy="814.553" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1415.4" cy="774.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1415.4" cy="774.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1245.45" cy="789.848" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1363.68" cy="815.453" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1363.68" cy="815.453" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1254.02" cy="843.479" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1409.02" cy="796.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1409.02" cy="796.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.05" cy="812.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.05" cy="812.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1332.05" cy="812.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.59" cy="751.865" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.59" cy="751.865" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.59" cy="751.865" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.59" cy="751.865" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1237.43" cy="865.039" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.34" cy="839.386" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1412.68" cy="782.143" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1412.68" cy="782.143" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.02" cy="811.59" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.02" cy="811.59" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.02" cy="811.59" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.15" cy="887.905" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.44" cy="798.028" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.44" cy="798.028" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.44" cy="798.028" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.91" cy="878.079" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.92" cy="836.131" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1301.53" cy="836.621" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.48" cy="746.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.48" cy="746.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.48" cy="746.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.48" cy="746.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1384.08" cy="812.68" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1384.08" cy="812.68" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.91" cy="835.231" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.91" cy="835.231" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1352.03" cy="801.526" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1352.03" cy="801.526" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1352.03" cy="801.526" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1284.34" cy="792.216" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1284.34" cy="792.216" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1284.34" cy="792.216" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1277.61" cy="830.729" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1343.84" cy="856.421" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.01" cy="792.427" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.01" cy="792.427" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.01" cy="792.427" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.01" cy="792.427" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.85" cy="916.948" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.45" cy="792.812" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.45" cy="792.812" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.53" cy="891.923" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.4" cy="786.173" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.4" cy="786.173" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.4" cy="786.173" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.4" cy="786.173" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1289.86" cy="805.57" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1289.86" cy="805.57" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.99" cy="791.484" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.99" cy="791.484" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.99" cy="791.484" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.99" cy="791.484" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.99" cy="791.484" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1271.04" cy="791.164" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.25" cy="842.128" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.25" cy="842.128" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1327.78" cy="859.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1287.78" cy="869.776" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1302.81" cy="858.179" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1441.3" cy="868.233" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1304.99" cy="754.396" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1304.99" cy="754.396" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1304.99" cy="754.396" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1304.99" cy="754.396" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1363.74" cy="850.216" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.8" cy="769.951" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.8" cy="769.951" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.8" cy="769.951" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.8" cy="769.951" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.8" cy="769.951" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.43" cy="824.114" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.43" cy="824.114" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1334.43" cy="824.114" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.14" cy="802.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.14" cy="802.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.14" cy="802.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.14" cy="802.472" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.29" cy="795.595" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.29" cy="795.595" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.29" cy="795.595" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.29" cy="795.595" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1288.91" cy="833.698" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.88" cy="823.785" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.88" cy="823.785" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.83" cy="750.248" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.59" cy="822.699" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.59" cy="822.699" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.59" cy="822.699" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.92" cy="741.14" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.92" cy="741.14" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.92" cy="741.14" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.92" cy="741.14" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.92" cy="741.14" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1387.88" cy="767.058" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1387.88" cy="767.058" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1387.88" cy="767.058" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1387.88" cy="767.058" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1294.57" cy="767.425" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1294.57" cy="767.425" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1294.57" cy="767.425" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1294.57" cy="767.425" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.26" cy="819.36" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.26" cy="819.36" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1280.65" cy="836.954" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.48" cy="855.391" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.91" cy="820.633" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.42" cy="805.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.42" cy="805.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.42" cy="805.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.42" cy="805.9" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.47" cy="800.837" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.47" cy="800.837" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.47" cy="800.837" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.47" cy="800.837" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.23" cy="775.863" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.23" cy="775.863" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.23" cy="775.863" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.23" cy="775.863" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1346.23" cy="775.863" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.02" cy="845.878" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.02" cy="845.878" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.8" cy="801.806" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.8" cy="801.806" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.8" cy="801.806" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.8" cy="801.806" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1338.99" cy="794.061" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1338.99" cy="794.061" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1338.99" cy="794.061" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1338.99" cy="794.061" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.3" cy="803.873" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.3" cy="803.873" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.3" cy="803.873" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.3" cy="803.873" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.41" cy="888.944" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1424.53" cy="819.435" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.08" cy="788.103" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.08" cy="788.103" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.08" cy="788.103" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.57" cy="865.311" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1411.82" cy="713.432" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1411.82" cy="713.432" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.7" cy="677.226" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1455.65" cy="837.334" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1404.6" cy="825.913" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.24" cy="770.218" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1328.63" cy="836.476" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.59" cy="783.631" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.59" cy="783.631" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.59" cy="783.631" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.59" cy="783.631" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1437.68" cy="713.255" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.16" cy="856.499" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1407.12" cy="779.055" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1407.12" cy="779.055" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.04" cy="778.779" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.04" cy="778.779" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.98" cy="898.515" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.96" cy="851.441" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1423.1" cy="710.421" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1390.04" cy="902.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.06" cy="863.193" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.62" cy="807.822" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.62" cy="807.822" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.9" cy="864.117" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1433.23" cy="798.607" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.72" cy="815.347" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1348.72" cy="815.347" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.6" cy="870.069" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1344.57" cy="865.809" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.91" cy="875.284" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1385.9" cy="797.448" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1385.9" cy="797.448" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.91" cy="874.269" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.56" cy="842.548" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.8" cy="827.83" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.8" cy="827.83" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1415.76" cy="838.8" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1383.39" cy="813.441" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1383.39" cy="813.441" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1263.55" cy="719.601" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.18" cy="756.824" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.18" cy="756.824" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.18" cy="756.824" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.18" cy="756.824" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1330.18" cy="756.824" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.02" cy="816.652" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.14" cy="704.997" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.14" cy="704.997" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.14" cy="704.997" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.87" cy="833.455" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.87" cy="833.455" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1340.04" cy="836.265" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1340.04" cy="836.265" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1413.86" cy="779.504" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.69" cy="853.18" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1239.59" cy="791.13" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.02" cy="827.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.02" cy="827.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.39" cy="796.728" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.39" cy="796.728" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.39" cy="796.728" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.39" cy="796.728" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.1" cy="799.604" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.1" cy="799.604" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.1" cy="799.604" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1316.1" cy="799.604" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.47" cy="803.564" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.47" cy="803.564" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.47" cy="803.564" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.47" cy="803.564" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.51" cy="846.313" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.49" cy="771.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.49" cy="771.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.49" cy="771.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.49" cy="771.555" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.69" cy="885.25" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.3" cy="859.407" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.71" cy="830.582" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.71" cy="830.582" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.98" cy="819.597" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.98" cy="819.597" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1414.42" cy="801.971" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1419.29" cy="805.843" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1273.19" cy="822.158" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.29" cy="850.163" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1283.62" cy="839.276" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.65" cy="811.598" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.65" cy="811.598" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.65" cy="811.598" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1440.36" cy="773.035" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1391.14" cy="837.456" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.4" cy="785.148" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.4" cy="785.148" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.4" cy="785.148" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.96" cy="748.664" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.93" cy="825.709" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.93" cy="825.709" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1357.93" cy="825.709" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.27" cy="810.941" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.27" cy="810.941" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.27" cy="810.941" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.84" cy="846.207" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1397.18" cy="852.541" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.28" cy="867.375" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1404.83" cy="814.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.15" cy="758.243" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1430.24" cy="727.48" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.14" cy="835.471" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.14" cy="835.471" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1390.42" cy="813.683" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1390.42" cy="813.683" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.17" cy="822.082" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.14" cy="897.934" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.2" cy="802.014" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.2" cy="802.014" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.88" cy="781.133" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.88" cy="781.133" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.88" cy="781.133" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1350.03" cy="814.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1350.03" cy="814.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1350.03" cy="814.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.79" cy="830.645" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1483.04" cy="805.753" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.66" cy="815.108" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.66" cy="815.108" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1416.94" cy="786.565" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1416.94" cy="786.565" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1409.52" cy="733.282" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.3" cy="916.963" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.46" cy="818.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.46" cy="818.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.46" cy="818.478" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.97" cy="816.125" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.97" cy="816.125" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.97" cy="816.125" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.03" cy="845.751" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.41" cy="818.202" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.41" cy="818.202" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1355.41" cy="818.202" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1414.62" cy="810.967" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.21" cy="774.871" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.21" cy="774.871" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.21" cy="774.871" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.21" cy="774.871" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1379.65" cy="829.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.64" cy="768.552" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.64" cy="768.552" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.64" cy="768.552" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1388.64" cy="768.552" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.63" cy="827" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.63" cy="827" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1394.1" cy="848.848" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1272.28" cy="739.971" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1272.28" cy="739.971" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.37" cy="847.645" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.23" cy="801.957" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.69" cy="881.602" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1377.52" cy="832.436" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.4" cy="815.65" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.11" cy="753.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.11" cy="753.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.62" cy="879.352" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1275.32" cy="798.18" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1432.33" cy="878.864" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.94" cy="820.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.94" cy="820.774" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.26" cy="829.673" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.26" cy="829.673" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.55" cy="732.551" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.55" cy="732.551" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.55" cy="732.551" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.55" cy="732.551" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.55" cy="732.551" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1265.66" cy="818.185" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.35" cy="838.195" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1359.48" cy="773.246" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.49" cy="825.745" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.25" cy="741.987" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1297.14" cy="689.932" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.24" cy="734.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.24" cy="734.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.24" cy="734.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.24" cy="734.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1336.24" cy="734.263" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1427.41" cy="769.64" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.15" cy="788.87" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.15" cy="788.87" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.15" cy="788.87" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1308.15" cy="788.87" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.18" cy="820.233" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.18" cy="820.233" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.72" cy="811.485" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.72" cy="811.485" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1386.72" cy="811.485" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.6" cy="762.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.6" cy="762.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.6" cy="762.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.6" cy="762.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1369.6" cy="762.558" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1413.93" cy="832.815" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.89" cy="835.052" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.25" cy="826.859" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.25" cy="826.859" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1394.83" cy="800.374" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1394.83" cy="800.374" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1394.83" cy="800.374" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1389.48" cy="828.115" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.77" cy="778.31" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.77" cy="778.31" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.77" cy="778.31" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.77" cy="778.31" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.77" cy="778.31" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.37" cy="880.336" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1440.64" cy="783.985" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1408.66" cy="687.885" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.05" cy="788.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.05" cy="788.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.05" cy="788.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.05" cy="788.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1342.05" cy="788.192" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.48" cy="828.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.26" cy="895.064" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.73" cy="813.643" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.73" cy="813.643" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.06" cy="728.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.06" cy="728.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.06" cy="728.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.06" cy="728.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1405.57" cy="842.056" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.6" cy="774.462" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.6" cy="774.462" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.6" cy="774.462" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.34" cy="808.049" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.34" cy="808.049" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.34" cy="808.049" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1270" cy="715.146" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1261.41" cy="815.341" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1340.21" cy="829.145" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1340.21" cy="829.145" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1263.69" cy="731.062" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1263.69" cy="731.062" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.8" cy="825.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1326.8" cy="825.22" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1393.93" cy="858.835" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1268.6" cy="836.357" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.94" cy="794.905" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.94" cy="794.905" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.94" cy="794.905" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.62" cy="811.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.62" cy="811.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1322.62" cy="811.974" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.91" cy="857.966" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1409.09" cy="823.661" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1337.39" cy="877.848" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1381.64" cy="831.008" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1402.5" cy="785.282" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1402.5" cy="785.282" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1402.5" cy="785.282" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1407.97" cy="743.222" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1407.97" cy="743.222" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1371.09" cy="851.569" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1367.49" cy="814.229" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1367.49" cy="814.229" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1367.49" cy="814.229" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.55" cy="824.438" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1375.55" cy="824.438" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.27" cy="782.382" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.27" cy="782.382" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.27" cy="782.382" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.27" cy="782.382" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.27" cy="782.382" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1417.75" cy="787.065" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1344.75" cy="847.816" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1344.75" cy="847.816" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.16" cy="762.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.16" cy="762.095" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.82" cy="823.543" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.82" cy="823.543" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.36" cy="772.915" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.65" cy="767.101" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.65" cy="767.101" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.65" cy="767.101" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1306.65" cy="767.101" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.62" cy="805.077" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.62" cy="805.077" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.62" cy="805.077" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1349.62" cy="805.077" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.38" cy="762.732" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.38" cy="762.732" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.38" cy="762.732" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1382.38" cy="762.732" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1292.59" cy="840.045" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.7" cy="848.438" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.29" cy="784.908" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.29" cy="784.908" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.29" cy="784.908" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1372.29" cy="784.908" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1341.59" cy="781.378" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.88" cy="809.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.88" cy="809.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.88" cy="809.19" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.32" cy="755.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.32" cy="755.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.32" cy="755.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.32" cy="755.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.32" cy="755.089" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1384.4" cy="888.085" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.59" cy="823.657" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1331.59" cy="823.657" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1274.28" cy="779.124" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1274.28" cy="779.124" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1305.05" cy="810.299" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1305.05" cy="810.299" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1305.05" cy="810.299" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.21" cy="822.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1324.21" cy="822.315" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1360.92" cy="837.677" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.92" cy="806.602" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.92" cy="806.602" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.92" cy="806.602" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.92" cy="806.602" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.8" cy="721.073" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1451.11" cy="817.49" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.54" cy="787.271" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.54" cy="787.271" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.41" cy="797.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.41" cy="797.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.41" cy="797.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1310.41" cy="797.111" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.74" cy="837.648" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.23" cy="838.541" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1435.28" cy="823.298" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1257.69" cy="806.024" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.16" cy="795.956" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.16" cy="795.956" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.16" cy="795.956" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.16" cy="795.956" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1298.56" cy="843.991" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.51" cy="812.21" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.51" cy="812.21" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1353.51" cy="812.21" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1356.01" cy="867.307" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1333.86" cy="840.435" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1329.17" cy="749.667" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1351.76" cy="821.312" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1351.76" cy="821.312" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1351.76" cy="821.312" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.18" cy="806.763" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.18" cy="806.763" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.18" cy="806.763" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1358.18" cy="806.763" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1317.86" cy="706.586" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1317.86" cy="706.586" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1317.86" cy="706.586" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1323.91" cy="836.711" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.35" cy="758.4" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.35" cy="758.4" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.35" cy="758.4" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.35" cy="758.4" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1366.35" cy="758.4" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1447.08" cy="821.84" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1420.95" cy="795.584" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1299.11" cy="859.016" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.28" cy="821.105" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.28" cy="821.105" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.62" cy="838.434" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1345.62" cy="838.434" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.22" cy="785.206" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.22" cy="785.206" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.74" cy="799.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.74" cy="799.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.74" cy="799.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1321.74" cy="799.303" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1392.31" cy="834.297" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1301.15" cy="831.128" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1301.15" cy="831.128" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.72" cy="777.359" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.72" cy="777.359" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.72" cy="777.359" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1361.72" cy="777.359" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.23" cy="789.433" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.23" cy="789.433" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.23" cy="789.433" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.23" cy="789.433" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.23" cy="789.433" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1320.44" cy="844.213" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.05" cy="806.397" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.05" cy="806.397" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1307.05" cy="806.397" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.11" cy="826.227" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1318.11" cy="826.227" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1290.98" cy="875.756" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.95" cy="765.646" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.95" cy="765.646" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1446.05" cy="744.399" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.52" cy="731.613" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1410.52" cy="731.613" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1374.6" cy="836.909" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.19" cy="806.015" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.19" cy="806.015" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.19" cy="806.015" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1347.19" cy="806.015" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1408.31" cy="836.759" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1269.64" cy="822.744" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1335.02" cy="866.501" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1266.84" cy="766.471" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1291.83" cy="817.993" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1291.83" cy="817.993" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1287.72" cy="834.93" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1403.2" cy="830.801" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370.02" cy="805.637" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370.02" cy="805.637" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1370.02" cy="805.637" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1406.55" cy="823.585" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.27" cy="801.127" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.27" cy="801.127" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1376.27" cy="801.127" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1297.32" cy="835.245" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1368.49" cy="831.242" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1368.49" cy="831.242" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.36" cy="811.386" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.36" cy="811.386" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1325.36" cy="811.386" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.85" cy="842.225" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1354.85" cy="842.225" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.26" cy="816.366" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1365.26" cy="816.366" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1270.66" cy="720.726" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1270.66" cy="720.726" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1396.28" cy="802.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1396.28" cy="802.43" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.38" cy="776.902" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.38" cy="776.902" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.38" cy="776.902" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.38" cy="776.902" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1362.38" cy="776.902" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.13" cy="829.6" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.13" cy="829.6" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1253.77" cy="800.408" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.39" cy="763.316" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.39" cy="763.316" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.39" cy="763.316" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1312.39" cy="763.316" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.73" cy="790.239" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.73" cy="790.239" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1398.73" cy="790.239" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.48" cy="817.93" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1319.48" cy="817.93" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.02" cy="842.447" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1339.02" cy="842.447" r="7" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<circle clip-path="url(#clip052)" cx="1364.16" cy="794.272" r="14" fill="#ff0000" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="3.2"/>
<path clip-path="url(#clip050)" d="
M1416.41 250.738 L1909.33 250.738 L1909.33 95.2176 L1416.41 95.2176  Z
  " fill="#ffffff" fill-rule="evenodd" fill-opacity="1"/>
<polyline clip-path="url(#clip050)" style="stroke:#000000; stroke-linecap:butt; stroke-linejoin:round; stroke-width:4; stroke-opacity:1; fill:none" points="
  1416.41,250.738 1909.33,250.738 1909.33,95.2176 1416.41,95.2176 1416.41,250.738 
  "/>
<circle clip-path="url(#clip050)" cx="1513.33" cy="147.058" r="23" fill="#000000" fill-rule="evenodd" fill-opacity="0.25" stroke="none"/>
<path clip-path="url(#clip050)" d="M1610.25 172.588 Q1612.94 172.588 1613.56 172.243 Q1614.2 171.921 1614.2 170.316 L1614.2 146.625 Q1614.2 144.945 1613.56 144.525 Q1612.94 144.106 1610.25 144.106 L1610.25 142.524 L1617.54 141.981 L1617.54 145.316 Q1620.58 141.981 1624.63 141.981 Q1628.93 141.981 1632.06 145.316 Q1635.2 148.651 1635.2 153.419 Q1635.2 158.236 1631.87 161.571 Q1628.53 164.906 1623.91 164.906 Q1621.39 164.906 1619.69 163.621 Q1618.01 162.312 1617.69 161.348 L1617.69 161.818 L1617.69 170.316 Q1617.69 171.946 1618.33 172.267 Q1618.97 172.588 1621.64 172.588 L1621.64 174.145 Q1616.33 173.997 1615.93 173.997 Q1615.44 173.997 1610.25 174.145 L1610.25 172.588 M1617.69 158.582 Q1617.69 159.347 1617.78 159.619 Q1617.91 159.891 1618.4 160.706 Q1620.4 163.769 1623.66 163.769 Q1623.69 163.769 1623.71 163.769 Q1626.6 163.769 1628.8 160.83 Q1631 157.865 1631 153.419 Q1631 149.17 1629 146.205 Q1627 143.241 1624.23 143.241 Q1622.26 143.241 1620.53 144.303 Q1618.8 145.365 1617.69 147.292 L1617.69 158.582 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1639.03 159.52 Q1639.03 155.222 1644.09 152.949 Q1647.13 151.492 1653.41 151.121 L1653.41 149.269 Q1653.41 146.131 1651.75 144.476 Q1650.12 142.796 1648.05 142.796 Q1644.34 142.796 1642.56 145.118 Q1644.07 145.168 1644.61 145.934 Q1645.16 146.675 1645.16 147.44 Q1645.16 148.453 1644.51 149.12 Q1643.9 149.763 1642.83 149.763 Q1641.82 149.763 1641.15 149.145 Q1640.49 148.503 1640.49 147.391 Q1640.49 144.921 1642.71 143.29 Q1644.96 141.66 1648.15 141.66 Q1652.3 141.66 1655.06 144.451 Q1655.93 145.316 1656.35 146.428 Q1656.79 147.539 1656.84 148.28 Q1656.89 148.997 1656.89 150.43 L1656.89 160.533 Q1656.89 160.83 1656.99 161.324 Q1657.09 161.818 1657.53 162.46 Q1658 163.078 1658.77 163.078 Q1660.6 163.078 1660.57 159.842 L1660.57 157.001 L1661.86 157.001 L1661.86 159.842 Q1661.86 162.534 1660.42 163.596 Q1659.01 164.634 1657.71 164.634 Q1656.03 164.634 1654.96 163.399 Q1653.9 162.164 1653.75 160.484 Q1652.99 162.411 1651.23 163.671 Q1649.5 164.906 1647.13 164.906 Q1645.3 164.906 1643.58 164.436 Q1641.87 163.992 1640.44 162.732 Q1639.03 161.447 1639.03 159.52 M1642.93 159.471 Q1642.93 161.348 1644.27 162.559 Q1645.6 163.769 1647.48 163.769 Q1649.6 163.769 1651.5 162.139 Q1653.41 160.484 1653.41 157.248 L1653.41 152.184 Q1647.8 152.381 1645.35 154.629 Q1642.93 156.852 1642.93 159.471 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1663.61 164.338 L1663.61 162.781 Q1666.3 162.781 1666.92 162.46 Q1667.56 162.114 1667.56 160.484 L1667.56 146.922 Q1667.56 145.044 1666.9 144.575 Q1666.25 144.106 1663.61 144.106 L1663.61 142.524 L1670.65 141.981 L1670.65 147.539 Q1671.37 145.365 1672.87 143.686 Q1674.4 141.981 1676.88 141.981 Q1678.51 141.981 1679.54 142.895 Q1680.61 143.809 1680.61 145.069 Q1680.61 146.181 1679.91 146.749 Q1679.25 147.292 1678.43 147.292 Q1677.52 147.292 1676.88 146.724 Q1676.26 146.131 1676.26 145.118 Q1676.26 144.501 1676.53 144.031 Q1676.83 143.537 1677.05 143.364 Q1677.27 143.191 1677.42 143.142 Q1677.32 143.093 1676.88 143.093 Q1674.08 143.093 1672.48 145.884 Q1670.9 148.651 1670.9 152.603 L1670.9 160.385 Q1670.9 161.843 1671.49 162.312 Q1672.11 162.781 1674.75 162.781 L1675.81 162.781 L1675.81 164.338 Q1673.79 164.189 1669.39 164.189 Q1668.77 164.189 1667.81 164.214 Q1666.85 164.239 1665.59 164.288 Q1664.33 164.338 1663.61 164.338 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1682.93 144.106 L1682.93 142.994 Q1685.25 142.895 1686.78 141.265 Q1688.34 139.609 1688.86 137.584 Q1689.4 135.558 1689.45 133.211 L1690.71 133.211 L1690.71 142.524 L1697.95 142.524 L1697.95 144.106 L1690.71 144.106 L1690.71 158.162 Q1690.71 163.621 1694.12 163.621 Q1695.58 163.621 1696.54 162.139 Q1697.5 160.632 1697.5 157.964 L1697.5 155.173 L1698.76 155.173 L1698.76 158.063 Q1698.76 160.805 1697.5 162.855 Q1696.24 164.906 1693.75 164.906 Q1692.83 164.906 1691.92 164.659 Q1691.03 164.436 1689.84 163.843 Q1688.68 163.226 1687.94 161.744 Q1687.23 160.237 1687.23 158.063 L1687.23 144.106 L1682.93 144.106 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1703.26 164.338 L1703.26 162.781 Q1705.95 162.781 1706.57 162.46 Q1707.19 162.114 1707.19 160.484 L1707.19 146.872 Q1707.19 144.995 1706.54 144.55 Q1705.93 144.106 1703.46 144.106 L1703.46 142.524 L1710.52 141.981 L1710.52 160.533 Q1710.52 162.04 1711.06 162.411 Q1711.61 162.781 1714.08 162.781 L1714.08 164.338 Q1708.92 164.189 1708.82 164.189 Q1708.13 164.189 1703.26 164.338 M1705.38 133.162 Q1705.38 132.149 1706.15 131.334 Q1706.94 130.494 1708.05 130.494 Q1709.16 130.494 1709.95 131.26 Q1710.74 132.001 1710.74 133.187 Q1710.74 134.348 1709.95 135.113 Q1709.16 135.855 1708.05 135.855 Q1706.89 135.855 1706.12 135.039 Q1705.38 134.224 1705.38 133.162 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1720.53 161.571 Q1717.31 158.211 1717.31 153.419 Q1717.31 148.602 1720.48 145.143 Q1723.64 141.66 1728.28 141.66 Q1731.37 141.66 1733.69 143.142 Q1736.02 144.6 1736.02 147.095 Q1736.02 148.206 1735.35 148.849 Q1734.71 149.466 1733.69 149.466 Q1732.63 149.466 1731.99 148.824 Q1731.37 148.157 1731.37 147.144 Q1731.37 146.699 1731.52 146.255 Q1731.67 145.81 1732.21 145.341 Q1732.78 144.871 1733.74 144.797 Q1731.94 142.944 1728.43 142.944 Q1728.38 142.944 1728.33 142.944 Q1725.76 142.944 1723.64 145.365 Q1721.51 147.786 1721.51 153.32 Q1721.51 156.21 1722.21 158.335 Q1722.92 160.434 1724.08 161.521 Q1725.24 162.608 1726.38 163.127 Q1727.52 163.621 1728.63 163.621 Q1733.59 163.621 1735.27 158.31 Q1735.42 157.816 1735.92 157.816 Q1736.58 157.816 1736.58 158.31 Q1736.58 158.557 1736.39 159.199 Q1736.19 159.842 1735.57 160.854 Q1734.95 161.867 1734.06 162.757 Q1733.2 163.621 1731.64 164.263 Q1730.11 164.906 1728.18 164.906 Q1723.74 164.906 1720.53 161.571 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1739.72 164.338 L1739.72 162.781 Q1742.41 162.781 1743.03 162.46 Q1743.65 162.114 1743.65 160.484 L1743.65 134.175 Q1743.65 132.297 1742.98 131.828 Q1742.34 131.359 1739.72 131.359 L1739.72 129.778 L1746.98 129.234 L1746.98 160.484 Q1746.98 162.114 1747.63 162.46 Q1748.27 162.781 1750.94 162.781 L1750.94 164.338 Q1750.32 164.338 1749.03 164.288 Q1747.75 164.239 1746.81 164.214 Q1745.87 164.189 1745.33 164.189 Q1744.74 164.189 1739.72 164.338 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1753.46 153.196 Q1753.46 148.429 1756.52 145.044 Q1759.58 141.66 1763.98 141.66 Q1768.43 141.66 1770.72 144.55 Q1773.05 147.44 1773.05 151.591 Q1773.05 152.356 1772.85 152.505 Q1772.65 152.653 1771.79 152.653 L1757.66 152.653 Q1757.66 157.816 1759.19 160.237 Q1761.31 163.621 1764.89 163.621 Q1765.39 163.621 1765.98 163.522 Q1766.57 163.424 1767.78 163.004 Q1768.99 162.559 1770.06 161.398 Q1771.12 160.237 1771.74 158.409 Q1771.88 157.692 1772.38 157.717 Q1773.05 157.717 1773.05 158.31 Q1773.05 158.755 1772.55 159.767 Q1772.08 160.756 1771.17 161.941 Q1770.25 163.127 1768.48 164.016 Q1766.72 164.906 1764.6 164.906 Q1760.15 164.906 1756.79 161.546 Q1753.46 158.162 1753.46 153.196 M1757.7 151.591 L1769.71 151.591 Q1769.71 150.528 1769.51 149.343 Q1769.34 148.157 1768.8 146.526 Q1768.28 144.871 1767.04 143.834 Q1765.81 142.796 1763.98 142.796 Q1763.16 142.796 1762.27 143.142 Q1761.41 143.488 1760.35 144.328 Q1759.29 145.168 1758.54 147.07 Q1757.8 148.972 1757.7 151.591 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1776.18 163.671 L1776.18 157.001 Q1776.18 156.432 1776.21 156.235 Q1776.23 156.037 1776.38 155.889 Q1776.53 155.741 1776.83 155.741 Q1777.17 155.741 1777.29 155.889 Q1777.44 156.037 1777.54 156.605 Q1778.31 160.089 1779.94 161.941 Q1781.59 163.769 1784.58 163.769 Q1787.42 163.769 1788.86 162.51 Q1790.29 161.25 1790.29 159.175 Q1790.29 155.469 1785.03 154.53 Q1781.99 153.913 1780.73 153.517 Q1779.47 153.098 1778.36 152.184 Q1776.18 150.405 1776.18 147.885 Q1776.18 145.365 1778.08 143.513 Q1780.01 141.66 1784.26 141.66 Q1787.1 141.66 1789.03 143.093 Q1789.6 142.648 1789.89 142.277 Q1790.56 141.66 1790.91 141.66 Q1791.3 141.66 1791.38 141.932 Q1791.45 142.179 1791.45 142.895 L1791.45 147.984 Q1791.45 148.552 1791.42 148.75 Q1791.4 148.947 1791.25 149.096 Q1791.1 149.219 1790.78 149.219 Q1790.21 149.219 1790.19 148.75 Q1789.79 142.623 1784.26 142.623 Q1781.27 142.623 1779.94 143.784 Q1778.6 144.921 1778.6 146.477 Q1778.6 147.342 1779 148.009 Q1779.42 148.651 1779.94 149.046 Q1780.48 149.417 1781.47 149.787 Q1782.46 150.133 1783.05 150.257 Q1783.67 150.38 1784.83 150.627 Q1788.88 151.393 1790.58 153.048 Q1792.71 155.173 1792.71 157.865 Q1792.71 160.854 1790.68 162.88 Q1788.66 164.906 1784.58 164.906 Q1781.3 164.906 1779 162.707 Q1778.7 163.004 1778.48 163.275 Q1778.26 163.522 1778.16 163.621 Q1778.08 163.72 1778.06 163.794 Q1778.04 163.843 1777.99 163.893 Q1776.97 164.906 1776.73 164.906 Q1776.33 164.906 1776.26 164.634 Q1776.18 164.387 1776.18 163.671 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><circle clip-path="url(#clip050)" cx="1513.33" cy="198.898" r="23" fill="#ff0000" fill-rule="evenodd" fill-opacity="1" stroke="#000000" stroke-opacity="1" stroke-width="5.12"/>
<path clip-path="url(#clip050)" d="M1613.59 213.46 Q1610.25 210.15 1610.25 205.357 Q1610.25 200.54 1613.51 197.032 Q1616.8 193.5 1621.49 193.5 Q1626.09 193.5 1629.37 197.008 Q1632.68 200.491 1632.68 205.357 Q1632.68 210.125 1629.35 213.435 Q1626.04 216.746 1621.44 216.746 Q1616.94 216.746 1613.59 213.46 M1614.45 204.938 Q1614.45 209.853 1615.73 212.077 Q1617.71 215.461 1621.49 215.461 Q1623.37 215.461 1624.92 214.448 Q1626.51 213.435 1627.37 211.731 Q1628.48 209.508 1628.48 204.938 Q1628.48 200.071 1627.15 197.922 Q1625.17 194.636 1621.44 194.636 Q1619.81 194.636 1618.2 195.501 Q1616.62 196.341 1615.66 198.021 Q1614.45 200.244 1614.45 204.938 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1635.55 183.199 L1635.55 181.618 L1642.83 181.074 L1642.83 197.107 Q1645.77 193.821 1649.78 193.821 Q1654.12 193.821 1657.31 197.156 Q1660.5 200.491 1660.5 205.259 Q1660.5 210.076 1657.16 213.411 Q1653.83 216.746 1649.21 216.746 Q1645.01 216.746 1642.59 213.04 Q1640.81 216.128 1640.76 216.178 L1639.5 216.178 L1639.5 186.015 Q1639.5 184.137 1638.83 183.668 Q1638.17 183.199 1635.55 183.199 M1642.98 210.422 Q1642.98 211.484 1643.7 212.546 Q1645.7 215.609 1648.96 215.609 Q1648.99 215.609 1649.01 215.609 Q1652.54 215.609 1654.74 212.373 Q1656.3 209.952 1656.3 205.209 Q1656.3 200.516 1654.84 198.169 Q1652.81 194.933 1649.53 194.933 Q1645.82 194.933 1643.55 198.169 Q1642.98 198.984 1642.98 199.997 L1642.98 210.422 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1663.88 215.511 L1663.88 208.841 Q1663.88 208.272 1663.91 208.075 Q1663.93 207.877 1664.08 207.729 Q1664.23 207.581 1664.52 207.581 Q1664.87 207.581 1664.99 207.729 Q1665.14 207.877 1665.24 208.445 Q1666.01 211.929 1667.64 213.781 Q1669.29 215.609 1672.28 215.609 Q1675.12 215.609 1676.55 214.35 Q1677.99 213.09 1677.99 211.015 Q1677.99 207.309 1672.73 206.37 Q1669.69 205.753 1668.43 205.357 Q1667.17 204.938 1666.06 204.024 Q1663.88 202.245 1663.88 199.725 Q1663.88 197.205 1665.78 195.353 Q1667.71 193.5 1671.96 193.5 Q1674.8 193.5 1676.73 194.933 Q1677.3 194.488 1677.59 194.117 Q1678.26 193.5 1678.6 193.5 Q1679 193.5 1679.07 193.772 Q1679.15 194.019 1679.15 194.735 L1679.15 199.824 Q1679.15 200.392 1679.12 200.59 Q1679.1 200.787 1678.95 200.936 Q1678.8 201.059 1678.48 201.059 Q1677.91 201.059 1677.89 200.59 Q1677.49 194.463 1671.96 194.463 Q1668.97 194.463 1667.64 195.624 Q1666.3 196.761 1666.3 198.317 Q1666.3 199.182 1666.7 199.849 Q1667.12 200.491 1667.64 200.886 Q1668.18 201.257 1669.17 201.627 Q1670.16 201.973 1670.75 202.097 Q1671.37 202.22 1672.53 202.467 Q1676.58 203.233 1678.28 204.888 Q1680.41 207.013 1680.41 209.705 Q1680.41 212.694 1678.38 214.72 Q1676.36 216.746 1672.28 216.746 Q1668.99 216.746 1666.7 214.547 Q1666.4 214.844 1666.18 215.115 Q1665.96 215.362 1665.86 215.461 Q1665.78 215.56 1665.76 215.634 Q1665.73 215.683 1665.68 215.733 Q1664.67 216.746 1664.42 216.746 Q1664.03 216.746 1663.96 216.474 Q1663.88 216.227 1663.88 215.511 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1683.52 205.036 Q1683.52 200.269 1686.58 196.884 Q1689.65 193.5 1694.04 193.5 Q1698.49 193.5 1700.79 196.39 Q1703.11 199.28 1703.11 203.431 Q1703.11 204.196 1702.91 204.345 Q1702.71 204.493 1701.85 204.493 L1687.72 204.493 Q1687.72 209.656 1689.25 212.077 Q1691.38 215.461 1694.96 215.461 Q1695.45 215.461 1696.05 215.362 Q1696.64 215.264 1697.85 214.844 Q1699.06 214.399 1700.12 213.238 Q1701.18 212.077 1701.8 210.249 Q1701.95 209.532 1702.44 209.557 Q1703.11 209.557 1703.11 210.15 Q1703.11 210.595 1702.62 211.607 Q1702.15 212.596 1701.23 213.781 Q1700.32 214.967 1698.54 215.856 Q1696.79 216.746 1694.66 216.746 Q1690.22 216.746 1686.86 213.386 Q1683.52 210.002 1683.52 205.036 M1687.77 203.431 L1699.78 203.431 Q1699.78 202.368 1699.58 201.183 Q1699.4 199.997 1698.86 198.366 Q1698.34 196.711 1697.11 195.674 Q1695.87 194.636 1694.04 194.636 Q1693.23 194.636 1692.34 194.982 Q1691.47 195.328 1690.41 196.168 Q1689.35 197.008 1688.61 198.91 Q1687.87 200.812 1687.77 203.431 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1705.98 216.178 L1705.98 214.621 Q1708.67 214.621 1709.29 214.3 Q1709.93 213.954 1709.93 212.324 L1709.93 198.762 Q1709.93 196.884 1709.26 196.415 Q1708.62 195.946 1705.98 195.946 L1705.98 194.364 L1713.02 193.821 L1713.02 199.379 Q1713.73 197.205 1715.24 195.526 Q1716.77 193.821 1719.24 193.821 Q1720.87 193.821 1721.91 194.735 Q1722.97 195.649 1722.97 196.909 Q1722.97 198.021 1722.28 198.589 Q1721.61 199.132 1720.8 199.132 Q1719.88 199.132 1719.24 198.564 Q1718.62 197.971 1718.62 196.958 Q1718.62 196.341 1718.9 195.871 Q1719.19 195.377 1719.41 195.204 Q1719.64 195.031 1719.79 194.982 Q1719.69 194.933 1719.24 194.933 Q1716.45 194.933 1714.84 197.724 Q1713.26 200.491 1713.26 204.443 L1713.26 212.225 Q1713.26 213.683 1713.86 214.152 Q1714.47 214.621 1717.12 214.621 L1718.18 214.621 L1718.18 216.178 Q1716.15 216.029 1711.76 216.029 Q1711.14 216.029 1710.18 216.054 Q1709.21 216.079 1707.95 216.128 Q1706.69 216.178 1705.98 216.178 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1725.29 195.946 L1725.29 194.364 Q1727.57 194.513 1730.21 194.513 Q1731.17 194.513 1735.72 194.364 L1735.72 195.946 Q1732.83 195.946 1732.83 197.255 Q1732.83 197.452 1733.12 198.119 L1738.81 212.225 L1743.97 199.33 Q1744.27 198.465 1744.27 198.119 Q1744.27 197.255 1743.67 196.637 Q1743.1 195.995 1741.85 195.946 L1741.85 194.364 Q1744.41 194.513 1746.24 194.513 Q1748.32 194.513 1750.02 194.364 L1750.02 195.946 Q1748.52 195.946 1747.45 196.637 Q1746.39 197.304 1746.04 197.897 Q1745.72 198.465 1745.38 199.33 L1738.76 215.782 Q1738.36 216.746 1737.69 216.746 Q1737.25 216.746 1737.05 216.548 Q1736.88 216.375 1736.63 215.782 L1729.35 197.823 Q1728.85 196.563 1728.21 196.267 Q1727.57 195.946 1725.29 195.946 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1751.7 211.36 Q1751.7 207.062 1756.77 204.789 Q1759.8 203.332 1766.08 202.961 L1766.08 201.109 Q1766.08 197.971 1764.42 196.316 Q1762.79 194.636 1760.72 194.636 Q1757.01 194.636 1755.23 196.958 Q1756.74 197.008 1757.28 197.774 Q1757.83 198.515 1757.83 199.28 Q1757.83 200.293 1757.19 200.96 Q1756.57 201.603 1755.51 201.603 Q1754.49 201.603 1753.83 200.985 Q1753.16 200.343 1753.16 199.231 Q1753.16 196.761 1755.38 195.13 Q1757.63 193.5 1760.82 193.5 Q1764.97 193.5 1767.73 196.291 Q1768.6 197.156 1769.02 198.268 Q1769.46 199.379 1769.51 200.12 Q1769.56 200.837 1769.56 202.27 L1769.56 212.373 Q1769.56 212.67 1769.66 213.164 Q1769.76 213.658 1770.2 214.3 Q1770.67 214.918 1771.44 214.918 Q1773.27 214.918 1773.24 211.682 L1773.24 208.841 L1774.53 208.841 L1774.53 211.682 Q1774.53 214.374 1773.09 215.436 Q1771.69 216.474 1770.38 216.474 Q1768.7 216.474 1767.64 215.239 Q1766.57 214.004 1766.42 212.324 Q1765.66 214.251 1763.91 215.511 Q1762.18 216.746 1759.8 216.746 Q1757.98 216.746 1756.25 216.276 Q1754.54 215.832 1753.11 214.572 Q1751.7 213.287 1751.7 211.36 M1755.6 211.311 Q1755.6 213.188 1756.94 214.399 Q1758.27 215.609 1760.15 215.609 Q1762.27 215.609 1764.18 213.979 Q1766.08 212.324 1766.08 209.088 L1766.08 204.024 Q1760.47 204.221 1758.03 206.469 Q1755.6 208.692 1755.6 211.311 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1775.84 195.946 L1775.84 194.834 Q1778.16 194.735 1779.69 193.105 Q1781.25 191.449 1781.77 189.424 Q1782.31 187.398 1782.36 185.051 L1783.62 185.051 L1783.62 194.364 L1790.86 194.364 L1790.86 195.946 L1783.62 195.946 L1783.62 210.002 Q1783.62 215.461 1787.03 215.461 Q1788.49 215.461 1789.45 213.979 Q1790.41 212.472 1790.41 209.804 L1790.41 207.013 L1791.67 207.013 L1791.67 209.903 Q1791.67 212.645 1790.41 214.695 Q1789.15 216.746 1786.66 216.746 Q1785.74 216.746 1784.83 216.499 Q1783.94 216.276 1782.75 215.683 Q1781.59 215.066 1780.85 213.584 Q1780.14 212.077 1780.14 209.903 L1780.14 195.946 L1775.84 195.946 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1796.17 216.178 L1796.17 214.621 Q1798.86 214.621 1799.48 214.3 Q1800.1 213.954 1800.1 212.324 L1800.1 198.712 Q1800.1 196.835 1799.45 196.39 Q1798.84 195.946 1796.37 195.946 L1796.37 194.364 L1803.43 193.821 L1803.43 212.373 Q1803.43 213.88 1803.97 214.251 Q1804.52 214.621 1806.99 214.621 L1806.99 216.178 Q1801.82 216.029 1801.73 216.029 Q1801.03 216.029 1796.17 216.178 M1798.29 185.002 Q1798.29 183.989 1799.06 183.174 Q1799.85 182.334 1800.96 182.334 Q1802.07 182.334 1802.86 183.1 Q1803.65 183.841 1803.65 185.027 Q1803.65 186.188 1802.86 186.953 Q1802.07 187.695 1800.96 187.695 Q1799.8 187.695 1799.03 186.879 Q1798.29 186.064 1798.29 185.002 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1813.24 213.46 Q1809.9 210.15 1809.9 205.357 Q1809.9 200.54 1813.16 197.032 Q1816.45 193.5 1821.14 193.5 Q1825.74 193.5 1829.02 197.008 Q1832.33 200.491 1832.33 205.357 Q1832.33 210.125 1829 213.435 Q1825.69 216.746 1821.09 216.746 Q1816.6 216.746 1813.24 213.46 M1814.1 204.938 Q1814.1 209.853 1815.39 212.077 Q1817.36 215.461 1821.14 215.461 Q1823.02 215.461 1824.58 214.448 Q1826.16 213.435 1827.02 211.731 Q1828.13 209.508 1828.13 204.938 Q1828.13 200.071 1826.8 197.922 Q1824.82 194.636 1821.09 194.636 Q1819.46 194.636 1817.86 195.501 Q1816.28 196.341 1815.31 198.021 Q1814.1 200.244 1814.1 204.938 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /><path clip-path="url(#clip050)" d="M1835.42 216.178 L1835.42 214.621 Q1838.11 214.621 1838.73 214.3 Q1839.35 213.954 1839.35 212.324 L1839.35 198.762 Q1839.35 196.884 1838.68 196.415 Q1838.04 195.946 1835.42 195.946 L1835.42 194.364 L1842.54 193.821 L1842.54 199.132 Q1844.91 193.821 1850.02 193.821 Q1853.83 193.821 1855.33 195.698 Q1856.44 196.958 1856.67 198.366 Q1856.91 199.775 1856.91 203.431 L1856.91 213.09 Q1856.96 214.152 1857.78 214.399 Q1858.59 214.621 1860.87 214.621 L1860.87 216.178 Q1855.7 216.029 1855.18 216.029 Q1854.79 216.029 1849.48 216.178 L1849.48 214.621 Q1852.17 214.621 1852.79 214.3 Q1853.43 213.954 1853.43 212.324 L1853.43 200.54 Q1853.43 197.971 1852.64 196.464 Q1851.85 194.933 1849.68 194.933 Q1847.06 194.933 1844.96 197.131 Q1842.86 199.33 1842.86 203.035 L1842.86 212.324 Q1842.86 213.954 1843.47 214.3 Q1844.12 214.621 1846.78 214.621 L1846.78 216.178 Q1841.62 216.029 1841.13 216.029 Q1840.73 216.029 1835.42 216.178 Z" fill="#000000" fill-rule="evenodd" fill-opacity="1" /></svg>
"""

# ╔═╡ 7077ea51-a27d-4e48-8b31-7669d44f0ec1
md"## 2D Random Walk POMDP"

# ╔═╡ e2ff6756-3425-4b55-a558-e4d5e9ae62aa
import Distributions: MvNormal

# ╔═╡ 66074347-982a-4634-93e7-07db0eb2f568
import RandomExtensions: Uniform

# ╔═╡ 1f196868-0fb9-4e91-8c1b-4ec9e64aab7f
md"""
##### Generative Model
Using a generative model means we don't have to explicitly define the state and observation spaces.
"""

# ╔═╡ 24f1c1b4-e590-4dcd-832f-b3540466d3ff
pomdp = QuickPOMDP(
	function gen(s, a, rng)
		x′ = clamp(s[1] + a[1] + randn(), -15, 15)
		y′ = clamp(s[2] + a[2] + randn(), -15, 15)
		s′ = (x′, y′)
		o = rand(observation(pomdp, a, s′))
		return (sp=s′, o=o, r=0)
	end,
	initialstate = NTuple{2}=>Uniform(-15:0.1:15),
	actions = NTuple{2}=>Uniform(-1:0.1:1),
	observation = (a, s′)->MvNormal([s′[1], s′[2]], [1 0; 0 1]),
	initialobs = [(0.0, 0.0)],
	render = function render(step)
		xₚ = [s[1] for s in particles(step.b)]
		yₚ = [s[2] for s in particles(step.b)]
		scatter(xₚ, yₚ, alpha=0.25, markersize=2, color="black", label="particles")
		scatter!([step.s[1]], [step.s[2]], color="red", label="observation")
		scatter!(aspect_ratio=:equal)
		xlims!(-15, 15)
		ylims!(-15, 15)
	end
);

# ╔═╡ 0e91db06-085e-4141-9bb6-ae3ef1965597
md"""
Using `QuickPOMDPs`, we defined the entire problem in 22 lines of code (including the rendering)!
"""

# ╔═╡ c660f8f8-128b-4cd6-9054-7f64b4032f7b
md"""
- The 2d **initial state** $(x,y)$ is sampled uniformly between -15 and 15.
- The 2d **action** $(dx, dy)$ is sampled uniformly between -1 and 1; applied as a change to the state.
- The **observations** are distributed according to a _multivariate Gaussian_ (or normal) distribution, with the means as the previous state and a variance of 1.
- The **initial observation** is deterministically at $(0,0)$.
- The **generative model** takes in the state $s$ and action $a$ and will apply the action (clamping to the boundaries), and return the next state $s^\prime$ and new observation $o$.
- The **rendering** plots the individual particles and the current observation.
"""

# ╔═╡ cd7681a6-d6a2-4614-a766-cc7656bc5181
md"## Using a Particle Filter as a Belief Updater"

# ╔═╡ 4a52afa4-ba92-4922-aa62-390c29ecf4ed
import POMDPSimulators: HistoryRecorder

# ╔═╡ d2ff96ed-cd77-49d9-8437-d51f83d1cd4d
import POMDPPolicies: RandomPolicy

# ╔═╡ f2a4cc9c-9361-43be-86f3-e83c3d4f1492
md"""
Now we set up a simulator that records the history for 200 time steps.
"""

# ╔═╡ 57f7b614-5d7c-4384-a90b-7bb13534aa1a
recorder = HistoryRecorder(max_steps=200);

# ╔═╡ adb54de3-19ad-4a14-93f8-c26c1ef1c512
md"""
We define how to update the our beliefs using the particle filter named the `BootstrapFilter`.$^2$
"""

# ╔═╡ 50ba37c4-48af-447a-af73-b0603bad71db
updater = BootstrapFilter(pomdp, 1000);

# ╔═╡ b42ae075-c276-4dc5-950f-8446a7f8793a
md"""
We will employ a random policy so we can focus on the state estimation portion of this POMDP.
"""

# ╔═╡ 18c1c66e-a8af-43c8-9be1-8a7afb695039
policy = RandomPolicy(pomdp);

# ╔═╡ d8c77da3-8d6d-463d-8b53-64bfba0b8612
md"""
The `simulate` function takes the following function signature:
"""

# ╔═╡ 0c9d6521-8b09-4fe9-9dec-00385f18416c
md"""
```julia
simulate(sim::Simulator,
         m::POMDP,
         p::Policy,
         u::Updater,
         b0=initialstate(m),
         s0=rand(b0))
```
"""

# ╔═╡ 1731872b-99ed-4342-a3c0-2d40cb4b035e
md"""
We run our "recorder" simulator and collect each step into the `history` variable.
"""

# ╔═╡ 62a1b5ce-a944-4bd1-9b41-39873cadced6
history = simulate(recorder,
	               pomdp,
	               policy,
	               updater,
				   initialstate(pomdp),
                   rand(initialobs(pomdp)))

# ╔═╡ 6193514c-0745-418a-b44b-98099bc6217d
md"""
Finally, we can step through the individual timesteps of the simulation and render the what the state estimation looks like when using a particle filter. Notice that the particle are initially spread uniformly before any belief updating occurs—but as soon as we step, the belief updater (i.e., the particle filter) does a good job of capturing a distribution over the true state.
"""

# ╔═╡ 8e85c80a-a83d-4821-a0e9-4971cd3335f6
@bind t Slider(1:length(history), default=1)

# ╔═╡ 814c8498-50c4-4aa1-9af2-fd4f97714d0f
render(history[t])

# ╔═╡ 528f817b-ace3-463f-857d-0e8f9e8aa184
md"""
## Animated GIF
We can use `makegif` from `POMDPGifs` to animate the history using our defined `render` function.
"""

# ╔═╡ 107d4441-b050-4e45-a762-2636341428d4
import POMDPGifs: makegif

# ╔═╡ 57d2cf73-0878-495e-beb1-1544728bdeb2
md"Create particle filter GIF? $(@bind create_gif CheckBox(false))"

# ╔═╡ 47769d8a-a50b-4b9d-82b3-716224611875
begin
	if create_gif
		!isdir("gifs") && mkdir("gifs") # create "gifs" directory
		makegif(pomdp, history; filename="gifs/particle_filter.gif", fps=4)
	end

	if isfile("./gifs/particle_filter.gif")
		LocalResource("./gifs/particle_filter.gif")
	end
end

# ╔═╡ a7f45204-4a9c-43fe-a0f1-1159dad8af5b
md"""
## References
1. Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray, "Algorithms for Decision Making", *MIT Press*, 2022. [https://algorithmsbook.com](https://algorithmsbook.com)

2. N. J. Gordon, D. J. Salmond, and A. F. M. Smith, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", *IEEE Radar and Signal Processing*, vol. 140, no. 2, pp. 107–113, 1993.

"""

# ╔═╡ 9a56f909-c0d4-4801-af7e-3cf73aebcb66
TableOfContents(title="Particle Filtering", depth=4)

# ╔═╡ d8ba4d75-ee62-402d-9f7c-42c11be651a9
md"""
---
"""

# ╔═╡ eeff68e1-7738-4cf3-b929-114f6ff11f4e
html"""
<script>
var section = 0;
var subsection = 0;
var headers = document.querySelectorAll('h2, h3');
for (var i=0; i < headers.length; i++) {
    var header = headers[i];
    var text = header.innerText;
    var original = header.getAttribute("text-original");
    if (original === null) {
        // Save original header text
        header.setAttribute("text-original", text);
    } else {
        // Replace with original text before adding section number
        text = header.getAttribute("text-original");
    }
    var numbering = "";
    switch (header.tagName) {
        case 'H2':
            section += 1;
            numbering = section + ".";
            subsection = 0;
            break;
        case 'H3':
            subsection += 1;
            numbering = section + "." + subsection;
            break;
    }
    header.innerText = numbering + " " + text;
};
</script>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
POMDPGifs = "7f35509c-0cb9-11e9-0708-2928828cdbb7"
POMDPPolicies = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
POMDPSimulators = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
ParticleFilters = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuickPOMDPs = "8af83fb2-a731-493c-9049-9e19dbce6165"
RandomExtensions = "fb686558-2515-59ef-acaa-46db3789a887"

[compat]
Distributions = "~0.24.18"
POMDPGifs = "~0.1.1"
POMDPPolicies = "~0.4.1"
POMDPSimulators = "~0.3.12"
POMDPs = "~0.9.3"
ParticleFilters = "~0.5.3"
Plots = "~1.21.3"
PlutoUI = "~0.7.9"
QuickPOMDPs = "~0.2.11"
RandomExtensions = "~0.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "f87e559f87a45bece9c9ed97458d3afe98b1ebb9"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.1.0"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BeliefUpdaters]]
deps = ["POMDPModelTools", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "7d4f9d57116796ae3fc768d195386b0a42b4a58d"
uuid = "8bb6e9a1-7d73-552c-a44a-e5dc5634aac4"
version = "0.2.2"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "182da592436e287758ded5be6e32c406de3a2e47"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d59e8320c2747553788e4fc42231489cc602fa50"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.1+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedTupleTools]]
git-tree-sha1 = "63831dcea5e11db1c0925efe5ef5fc01d528c522"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.13.7"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[POMDPGifs]]
deps = ["POMDPModelTools", "POMDPPolicies", "POMDPSimulators", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reel"]
git-tree-sha1 = "34dc3a48236be73f8a8fd382307ff6db20379799"
uuid = "7f35509c-0cb9-11e9-0708-2928828cdbb7"
version = "0.1.1"

[[POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[POMDPModelTools]]
deps = ["CommonRLInterface", "Distributions", "LinearAlgebra", "POMDPLinter", "POMDPs", "Random", "SparseArrays", "Statistics", "Tricks", "UnicodePlots"]
git-tree-sha1 = "be6e420779e4a076acac228aa68440ae7ce73331"
uuid = "08074719-1b2a-587c-a292-00f91cc44415"
version = "0.3.7"

[[POMDPPolicies]]
deps = ["BeliefUpdaters", "Distributions", "LinearAlgebra", "POMDPModelTools", "POMDPs", "Parameters", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "2920bc20706b82cf6c5058da51b1bb5d3c391a27"
uuid = "182e52fb-cfd0-5e46-8c26-fd0667c990f4"
version = "0.4.1"

[[POMDPSimulators]]
deps = ["BeliefUpdaters", "DataFrames", "Distributed", "NamedTupleTools", "POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "ProgressMeter", "Random"]
git-tree-sha1 = "1c8a996d3b03023bdeb7589ad87231e73ba93e19"
uuid = "e0d0a172-29c6-5d4e-96d0-f262df5d01fd"
version = "0.3.12"

[[POMDPTesting]]
deps = ["POMDPs", "Random"]
git-tree-sha1 = "6186037fc901d91703c0aa7ab10c145eeb6d0796"
uuid = "92e6a534-49c2-5324-9027-86e3c861ab81"
version = "0.2.5"

[[POMDPs]]
deps = ["Distributions", "LightGraphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "3a8f6cf6a3b7b499ec4294f2eb2b16b9dc8a7513"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.3"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[ParticleFilters]]
deps = ["POMDPLinter", "POMDPModelTools", "POMDPPolicies", "POMDPs", "Random", "Statistics", "StatsBase"]
git-tree-sha1 = "9cdc1db2a4992d1ba19bf896372b4eaaac78fa98"
uuid = "c8b314e2-9260-5cf8-ae76-3be7461ca6d0"
version = "0.5.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9ff1c70190c1c30aebca35dc489f7411b256cd23"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.13"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "2dbafeadadcf7dadff20cd60046bba416b4912be"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.21.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[QuickPOMDPs]]
deps = ["BeliefUpdaters", "NamedTupleTools", "POMDPModelTools", "POMDPTesting", "POMDPs", "Random", "Tricks", "UUIDs"]
git-tree-sha1 = "f1819fb42ce01ab846b4e4a81a0509f3c2e80e24"
uuid = "8af83fb2-a731-493c-9049-9e19dbce6165"
version = "0.2.11"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "d4491becdc53580c6dadb0f6249f90caae888554"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.0"

[[Reel]]
deps = ["FFMPEG"]
git-tree-sha1 = "0f600c38899603d9667111176eb6b5b33c80781e"
uuid = "71555da5-176e-5e73-a222-aebc6c6e4f2f"
version = "1.3.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "1700b86ad59348c0f9f68ddc95117071f947072d"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tricks]]
git-tree-sha1 = "8280b6d0096e88b77a84f843fa18620d3b20e052"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.4"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodePlots]]
deps = ["Crayons", "Dates", "SparseArrays", "StatsBase"]
git-tree-sha1 = "dc9c7086d41783f14d215ea0ddcca8037a8691e9"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "1.4.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─5771f1a3-a0df-4d65-9e7d-4ab45fa57360
# ╟─19380e99-07e6-4490-a5e6-a451dde2e593
# ╟─96fe0aec-53e3-4aa9-971c-af6e4fa16d95
# ╟─7077ea51-a27d-4e48-8b31-7669d44f0ec1
# ╠═7197cded-3ba1-4d0b-81b8-cd20130e3ad5
# ╠═e2ff6756-3425-4b55-a558-e4d5e9ae62aa
# ╠═66074347-982a-4634-93e7-07db0eb2f568
# ╠═afcad738-b323-4f31-a84b-ea8d4d9d3937
# ╟─1f196868-0fb9-4e91-8c1b-4ec9e64aab7f
# ╠═24f1c1b4-e590-4dcd-832f-b3540466d3ff
# ╟─0e91db06-085e-4141-9bb6-ae3ef1965597
# ╟─c660f8f8-128b-4cd6-9054-7f64b4032f7b
# ╟─cd7681a6-d6a2-4614-a766-cc7656bc5181
# ╠═08f2254b-d7b6-46e2-8176-35e3c5561f97
# ╠═4a52afa4-ba92-4922-aa62-390c29ecf4ed
# ╠═d2ff96ed-cd77-49d9-8437-d51f83d1cd4d
# ╟─f2a4cc9c-9361-43be-86f3-e83c3d4f1492
# ╠═57f7b614-5d7c-4384-a90b-7bb13534aa1a
# ╟─adb54de3-19ad-4a14-93f8-c26c1ef1c512
# ╠═50ba37c4-48af-447a-af73-b0603bad71db
# ╟─b42ae075-c276-4dc5-950f-8446a7f8793a
# ╠═18c1c66e-a8af-43c8-9be1-8a7afb695039
# ╟─d8c77da3-8d6d-463d-8b53-64bfba0b8612
# ╟─0c9d6521-8b09-4fe9-9dec-00385f18416c
# ╟─1731872b-99ed-4342-a3c0-2d40cb4b035e
# ╠═62a1b5ce-a944-4bd1-9b41-39873cadced6
# ╟─6193514c-0745-418a-b44b-98099bc6217d
# ╠═814c8498-50c4-4aa1-9af2-fd4f97714d0f
# ╠═8e85c80a-a83d-4821-a0e9-4971cd3335f6
# ╟─528f817b-ace3-463f-857d-0e8f9e8aa184
# ╠═107d4441-b050-4e45-a762-2636341428d4
# ╟─57d2cf73-0878-495e-beb1-1544728bdeb2
# ╠═47769d8a-a50b-4b9d-82b3-716224611875
# ╟─a7f45204-4a9c-43fe-a0f1-1159dad8af5b
# ╠═9a56f909-c0d4-4801-af7e-3cf73aebcb66
# ╟─d8ba4d75-ee62-402d-9f7c-42c11be651a9
# ╟─eeff68e1-7738-4cf3-b929-114f6ff11f4e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
