import * as React from "react";
import { motion } from "framer-motion";
const Client = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="100%"
    height="100%"
    viewBox="0 0 402.8 209.124"
    className="py-10 invert"
    {...props}
  >
    <symbol
      id="image-e8fb129aa5a0d301a529101813663dcb80718e99"
      viewBox="0 0 400.8 800.124"
    >
      <image
        width="100%"
        height="100%"
        className="z-50 invert animate-bounce"
        href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjUwMCIgdmlld0JveD0iMTc1LjcgNzggNDkwLjYgNDM2LjkiIHdpZHRoPSIyMTk0Ij48ZyBmaWxsPSIjNjFkYWZiIj48cGF0aCBkPSJtNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2Ljl2LTIyLjNjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZ2LTIyLjNjLTguNCAwLTE2IDEuOC0yMi42IDUuNi0yOC4xIDE2LjItMzQuNCA2Ni43LTE5LjkgMTMwLjEtNjIuMiAxOS4yLTEwMi43IDQ5LjktMTAyLjcgODIuMyAwIDMyLjUgNDAuNyA2My4zIDEwMy4xIDgyLjQtMTQuNCA2My42LTggMTE0LjIgMjAuMiAxMzAuNCA2LjUgMy44IDE0LjEgNS42IDIyLjUgNS42IDI3LjUgMCA2My41LTE5LjYgOTkuOS01My42IDM2LjQgMzMuOCA3Mi40IDUzLjIgOTkuOSA1My4yIDguNCAwIDE2LTEuOCAyMi42LTUuNiAyOC4xLTE2LjIgMzQuNC02Ni43IDE5LjktMTMwLjEgNjItMTkuMSAxMDIuNS00OS45IDEwMi41LTgyLjN6bS0xMzAuMi02Ni43Yy0zLjcgMTIuOS04LjMgMjYuMi0xMy41IDM5LjUtNC4xLTgtOC40LTE2LTEzLjEtMjQtNC42LTgtOS41LTE1LjgtMTQuNC0yMy40IDE0LjIgMi4xIDI3LjkgNC43IDQxIDcuOXptLTQ1LjggMTA2LjVjLTcuOCAxMy41LTE1LjggMjYuMy0yNC4xIDM4LjItMTQuOSAxLjMtMzAgMi00NS4yIDItMTUuMSAwLTMwLjItLjctNDUtMS45LTguMy0xMS45LTE2LjQtMjQuNi0yNC4yLTM4LTcuNi0xMy4xLTE0LjUtMjYuNC0yMC44LTM5LjggNi4yLTEzLjQgMTMuMi0yNi44IDIwLjctMzkuOSA3LjgtMTMuNSAxNS44LTI2LjMgMjQuMS0zOC4yIDE0LjktMS4zIDMwLTIgNDUuMi0yIDE1LjEgMCAzMC4yLjcgNDUgMS45IDguMyAxMS45IDE2LjQgMjQuNiAyNC4yIDM4IDcuNiAxMy4xIDE0LjUgMjYuNCAyMC44IDM5LjgtNi4zIDEzLjQtMTMuMiAyNi44LTIwLjcgMzkuOXptMzIuMy0xM2M1LjQgMTMuNCAxMCAyNi44IDEzLjggMzkuOC0xMy4xIDMuMi0yNi45IDUuOS00MS4yIDggNC45LTcuNyA5LjgtMTUuNiAxNC40LTIzLjcgNC42LTggOC45LTE2LjEgMTMtMjQuMXptLTEwMS40IDEwNi43Yy05LjMtOS42LTE4LjYtMjAuMy0yNy44LTMyIDkgLjQgMTguMi43IDI3LjUuNyA5LjQgMCAxOC43LS4yIDI3LjgtLjctOSAxMS43LTE4LjMgMjIuNC0yNy41IDMyem0tNzQuNC01OC45Yy0xNC4yLTIuMS0yNy45LTQuNy00MS03LjkgMy43LTEyLjkgOC4zLTI2LjIgMTMuNS0zOS41IDQuMSA4IDguNCAxNiAxMy4xIDI0czkuNSAxNS44IDE0LjQgMjMuNHptNzMuOS0yMDguMWM5LjMgOS42IDE4LjYgMjAuMyAyNy44IDMyLTktLjQtMTguMi0uNy0yNy41LS43LTkuNCAwLTE4LjcuMi0yNy44LjcgOS0xMS43IDE4LjMtMjIuNCAyNy41LTMyem0tNzQgNTguOWMtNC45IDcuNy05LjggMTUuNi0xNC40IDIzLjctNC42IDgtOC45IDE2LTEzIDI0LTUuNC0xMy40LTEwLTI2LjgtMTMuOC0zOS44IDEzLjEtMy4xIDI2LjktNS44IDQxLjItNy45em0tOTAuNSAxMjUuMmMtMzUuNC0xNS4xLTU4LjMtMzQuOS01OC4zLTUwLjZzMjIuOS0zNS42IDU4LjMtNTAuNmM4LjYtMy43IDE4LTcgMjcuNy0xMC4xIDUuNyAxOS42IDEzLjIgNDAgMjIuNSA2MC45LTkuMiAyMC44LTE2LjYgNDEuMS0yMi4yIDYwLjYtOS45LTMuMS0xOS4zLTYuNS0yOC0xMC4yem01My44IDE0Mi45Yy0xMy42LTcuOC0xOS41LTM3LjUtMTQuOS03NS43IDEuMS05LjQgMi45LTE5LjMgNS4xLTI5LjQgMTkuNiA0LjggNDEgOC41IDYzLjUgMTAuOSAxMy41IDE4LjUgMjcuNSAzNS4zIDQxLjYgNTAtMzIuNiAzMC4zLTYzLjIgNDYuOS04NCA0Ni45LTQuNS0uMS04LjMtMS0xMS4zLTIuN3ptMjM3LjItNzYuMmM0LjcgMzguMi0xLjEgNjcuOS0xNC42IDc1LjgtMyAxLjgtNi45IDIuNi0xMS41IDIuNi0yMC43IDAtNTEuNC0xNi41LTg0LTQ2LjYgMTQtMTQuNyAyOC0zMS40IDQxLjMtNDkuOSAyMi42LTIuNCA0NC02LjEgNjMuNi0xMSAyLjMgMTAuMSA0LjEgMTkuOCA1LjIgMjkuMXptMzguNS02Ni43Yy04LjYgMy43LTE4IDctMjcuNyAxMC4xLTUuNy0xOS42LTEzLjItNDAtMjIuNS02MC45IDkuMi0yMC44IDE2LjYtNDEuMSAyMi4yLTYwLjYgOS45IDMuMSAxOS4zIDYuNSAyOC4xIDEwLjIgMzUuNCAxNS4xIDU4LjMgMzQuOSA1OC4zIDUwLjYtLjEgMTUuNy0yMyAzNS42LTU4LjQgNTAuNnoiLz48Y2lyY2xlIGN4PSI0MjAuOSIgY3k9IjI5Ni41IiByPSI0NS43Ii8+PC9nPjwvc3ZnPg=="
      />
    </symbol>
    <defs>
      <style>
        {
          '@font-face{font-family:"Virgil";src:url(https://excalidraw.com/Virgil.woff2)}@font-face{font-family:"Cascadia";src:url(https://excalidraw.com/Cascadia.woff2)}@font-face{font-family:"Assistant";src:url(https://excalidraw.com/Assistant-Regular.woff2)}'
        }
      </style>
    </defs>
    <path fill="#fff" d="M0 0h402.8v209.124H0z" />
    <g strokeLinecap="round">
      <path
        fill="none"
        stroke="#e7f5ff"
        strokeWidth={0.5}
        d="M18.63 59.312s0 0 0 0m0 0s0 0 0 0m-6.82 13.94c6.68-5.31 3.92-10.51 13.77-15.85m-13.77 15.85c6.2-4.09 11.08-12.78 13.77-15.85m-14.03 22.25c8.66-7.59 13.59-20.33 20.33-23.4m-20.33 23.4c8.23-9.57 14.54-18.19 20.33-23.4m-20.6 29.8c9.21-10.85 16.59-23.87 26.25-30.19m-26.25 30.19c7.98-6.2 11.19-12.64 26.25-30.19m-26.51 36.58c14.24-16.71 27.38-32.25 31.49-36.22m-31.49 36.22c8.8-8.13 14.84-16.8 31.49-36.22m-31.75 42.62c14.09-11.67 18.47-24.96 36.74-42.26m-36.74 42.26c10.82-12.32 24.17-27.02 36.74-42.26m-37 48.66c20.34-17.88 34.57-35.61 42.64-49.06m-42.64 49.06c10.87-13.77 22.75-26.7 42.64-49.06m-42.25 54.7c12.85-16.04 27.24-33.91 47.24-54.34m-47.24 54.34c13.86-17.79 29.07-33.17 47.24-54.34m-47.5 60.74c14.85-19.68 28.36-36.37 53.14-61.13m-53.14 61.13c17.34-17.96 35.02-40.54 53.14-61.13m-53.4 67.53c17.42-19.62 33.12-41.8 58.39-67.17m-58.39 67.17c24.99-24.61 47.6-51.94 58.39-67.17m-57.34 72.06c17.41-16.75 29.47-32.71 62.98-72.46m-62.98 72.46c18.11-18.56 33.76-38.65 62.98-72.46m-61.93 77.34c21.68-23.42 50.37-54.83 66.92-76.98m-66.92 76.98c17.92-21.33 36.8-43.94 66.92-76.98m-65.87 81.87c30.32-27.69 53.67-59.2 70.86-81.51m-70.86 81.51c20.95-27.36 44.12-51.28 70.86-81.51m-67.84 84.14c30.72-34.69 60.58-65.65 73.48-84.53m-73.48 84.53c24.83-26.36 50.11-55.56 73.48-84.53m-71.12 87.91c28.8-25.5 48.69-58.29 75.45-86.8m-75.45 86.8c28.55-34.64 59.05-70.07 75.45-86.8m-71.11 87.91c13.49-23.34 36.16-37.89 75.44-86.79m-75.44 86.79c24.07-30.27 50.09-59.03 75.44-86.79m-71.11 87.91c30.26-33.12 55.09-64.07 73.48-84.53m-73.48 84.53c13.52-16.94 30.21-34.57 73.48-84.53m-68.5 84.89c26.84-35.42 54.04-61.11 71.51-82.27m-71.51 82.27c27.08-27.34 51.18-59.63 71.51-82.27m-65.86 81.87c28.34-24.56 51.88-58.76 66.91-76.98m-66.91 76.98c16.38-19.27 30.16-34.6 66.91-76.98m-61.93 77.34c15.23-23.08 34.52-41.47 62.99-72.45m-62.99 72.45c19.67-20.62 35.28-43.15 62.99-72.45m-57.34 72.06c16.23-20.98 36.51-46.15 59.04-67.93m-59.04 67.93c15.68-18.28 30.4-33.2 59.04-67.93m-54.06 68.29c21.08-15.77 33.38-34.93 53.8-61.89m-53.8 61.89c18.29-19.49 34.84-40.28 53.8-61.89m-48.81 62.25c14.84-13.01 28.7-32.56 48.55-55.85m-48.55 55.85c10.01-14.61 23.9-28.55 48.55-55.85m-42.91 55.45c13.88-17.76 37.08-39.07 43.3-49.81m-43.3 49.81c11.12-9.37 17.26-21.6 43.3-49.81m-38.31 50.17c15.93-16.45 28.15-27.41 38.05-43.77m-38.05 43.77c10.3-13.34 22.39-26.9 38.05-43.77m-32.41 43.38c7.23-9.91 18.99-18.1 32.15-36.98m-32.15 36.98c11.32-8.77 19.74-22.97 32.15-36.98m-27.16 37.34c5.75-8.68 11.93-14.6 27.56-31.7m-27.56 31.7c10.49-12.47 18.3-23.29 27.56-31.7m-22.57 32.06c9.29-7.87 14.61-18.07 22.3-25.66m-22.3 25.66c7.17-7.14 9.55-14.74 22.3-25.66m-16 24.51c8.49-4.61 7.96-15.02 15.74-18.11m-15.74 18.11c5.45-6.2 10.64-11.79 15.74-18.11m-8.13 15.45c.55-3.22 1.36-3.44 5.9-6.79m-5.9 6.79c2.39-1.57 2.97-4.28 5.9-6.79"
      />
      <path
        fill="none"
        stroke="#1e1e1e"
        strokeDasharray="1.5 7"
        strokeWidth={1.5}
        d="M32.8 55.782c13.18 1.17 22.08 4.2 51.61 0m0 0c12.19-1.23 20.47 5.59 22.8 22.8m0 0c-3.03 14.67-2.89 31.27 0 45.6m0 0c1.06 17.75-5.65 23.91-22.8 22.8m0 0c-12.31-2.48-25.48-3.33-51.61 0m0 0c-12.84 2.92-20.55-3.89-22.8-22.8m0 0c.04-8.8.25-14.78 0-45.6m0 0c-3.2-12.58 7.17-20.63 22.8-22.8"
      />
    </g>
    <use
      width={50}
      height={57}
      href="#image-e8fb129aa5a0d301a529101813663dcb80718e99"
      transform="translate(35.066 71.9)"
    />
    <g strokeLinecap="round">
      <path
        fill="none"
        stroke="#ffc9c9"
        strokeWidth={0.5}
        d="M188.684 24.291s0 0 0 0m0 0s0 0 0 0m-.91 7.16c.81-1.69 4.43-6.75 7.21-8.31m-7.21 8.31c2.34-2.24 3.42-3.34 7.21-8.31m-7.48 14.7c3.83-7.39 5.35-8.84 13.13-15.09m-13.13 15.09c4.07-4.02 6-8.57 13.13-15.09m-13.39 21.49c6.15-8.01 4.55-9.88 18.37-21.13m-18.37 21.13c4.28-4.61 9.28-12.38 18.37-21.13m-17.97 26.77c6.27-8.19 9.28-16.9 22.96-26.41m-22.96 26.41c4.98-10.01 15.2-15.07 22.96-26.41m-23.22 32.81c9.38-11.3 11.83-15.83 25.58-29.43m-25.58 29.43c7.9-13.48 18.71-24.42 25.58-29.43m-25.19 35.07c5.64-9.91 12.8-17.76 27.56-31.69m-27.56 31.69c3.88-8.95 9.74-15.2 27.56-31.69m-27.82 38.09c8.86-6.95 14.88-19.5 27.55-31.7m-27.55 31.7c7.68-10.47 14.76-19.98 27.55-31.7m-27.81 38.1c7.1-13.1 17.96-18.71 28.21-32.45m-28.21 32.45c9.32-11.81 21.06-23.83 28.21-32.45m-27.82 38.09c7.98-9.91 13.21-16.77 27.56-31.69m-27.56 31.69c6.66-10 17.37-18.91 27.56-31.69m-27.82 38.09c12.17-9.38 19-28.65 28.21-32.45m-28.21 32.45c6.9-9.33 17.75-16.42 28.21-32.45m-27.81 38.09c2.05-3.53 7.98-14.47 27.55-31.69m-27.55 31.69c6.08-9.41 13.32-15.28 27.55-31.69m-27.82 38.09c8.16-10.55 11.92-20.1 27.56-31.7m-27.56 31.7c6.91-9.74 15.15-14.99 27.56-31.7m-27.82 38.1c12.45-9.13 22-20.97 28.21-32.45m-28.21 32.45c8.86-9.9 15.52-16.4 28.21-32.45m-27.81 38.09c10.65-6.4 21.01-18.52 27.55-31.7m-27.55 31.7c9.62-13.68 21.92-24.32 27.55-31.7m-27.81 38.1c2.93-5.78 15-10.32 28.21-32.45m-28.21 32.45c5.64-8.04 12.46-15.3 28.21-32.45m-27.82 38.09c6.06-6.76 11.64-11.81 27.56-31.69m-27.56 31.69c11.3-12.44 21.44-25.44 27.56-31.69m-27.82 38.09c10.62-11.55 12.49-16.17 27.55-31.7m-27.55 31.7c10.07-12.94 18.57-22.44 27.55-31.7m-27.81 38.1c10.47-7.9 16.02-23.31 28.21-32.45m-28.21 32.45c9.41-7.12 16.83-16.47 28.21-32.45m-27.82 38.09c12.77-11.59 17.79-25.65 27.56-31.7m-27.56 31.7c12.15-13.05 19.81-24.54 27.56-31.7m-27.82 38.1c9.28-9.19 14.13-14.01 28.21-32.45m-28.21 32.45c8.66-11.35 17.73-20.69 28.21-32.45m-27.81 38.09c6.38-12.51 15.08-17.22 27.55-31.69m-27.55 31.69c12.01-13.78 22.32-25.14 27.55-31.69m-27.82 38.09c7.23-9.1 18.66-25.21 27.56-31.7m-27.56 31.7c6.03-8.83 10.42-13.15 27.56-31.7m-27.82 38.1c8.39-6.33 10.23-13.37 28.21-32.45m-28.21 32.45c6.38-9.29 13.53-15.85 28.21-32.45m-27.81 38.09c7.26-7.85 11.66-19.4 27.55-31.7m-27.55 31.7c11.17-11.96 21.16-24.15 27.55-31.7m-27.81 38.1c11.13-13.88 20-20.77 28.21-32.45m-28.21 32.45c10.26-13.74 22.82-22.62 28.21-32.45m-27.82 38.09c6.25-5.22 15.91-16.24 27.55-31.7m-27.55 31.7c10.28-8.9 18.78-22.16 27.55-31.7m-25.84 35.84c4.62-3.44 11.39-16.43 25.58-29.44m-25.58 29.44c9.72-8.47 17.8-21.67 25.58-29.44m-21.91 31.31c9.27-10.25 14.12-19.71 22.31-25.66m-22.31 25.66c6.07-5.22 8.94-11.53 22.31-25.66m-16.67 25.26c7.33-8.92 7.42-8.88 16.41-18.87m-16.41 18.87c4.9-5.18 8.38-10.58 16.41-18.87m-11.42 19.23c3.16-4.78 4.31-6.57 11.81-13.58m-11.81 13.58c3.64-3.32 5.74-7.83 11.81-13.58m-6.16 13.19c1.5-2.77 2.79-2.27 5.24-6.04m-5.24 6.04c1.09-1.84 2.33-2.64 5.24-6.04"
      />
      <path
        fill="none"
        stroke="#1e1e1e"
        strokeDasharray="1.5 7"
        strokeWidth={1.5}
        d="M193.774 22.691c2.64-.36 8.59-1.39 13.85 0m0 0c1.38-2.02 8.65-.23 6.92 6.93m0 0c-3.93 52.48 2.45 96.71 0 153.42m0 0c-1.97 1.06 1.54 9.76-6.92 6.92m0 0c-3.36-2.38-9.3-1.57-13.85 0m0 0c-1.52-3.3-4.97-4.43-6.93-6.92m0 0c-3.04-36.8-1.31-68.42 0-153.42m0 0c-.51-8.41-.63-9.93 6.93-6.93"
      />
    </g>
    <text
      y={14.016}
      fill="#f08c00"
      dominantBaseline="alphabetic"
      fontFamily="Virgil, Segoe UI Emoji"
      fontSize={16}
      style={{
        whiteSpace: "pre",
      }}
      transform="translate(194.162 74.442)"
    >
      {"A"}
    </text>
    <text
      y={34.016}
      fill="#f08c00"
      dominantBaseline="alphabetic"
      fontFamily="Virgil, Segoe UI Emoji"
      fontSize={16}
      style={{
        whiteSpace: "pre",
      }}
      transform="translate(194.162 74.442)"
    >
      {"P "}
    </text>
    <text
      y={54.016}
      fill="#f08c00"
      dominantBaseline="alphabetic"
      fontFamily="Virgil, Segoe UI Emoji"
      fontSize={16}
      style={{
        whiteSpace: "pre",
      }}
      transform="translate(194.162 74.442)"
    >
      {"I"}
    </text>
    <g strokeLinecap="round">
      <path
        fill="none"
        stroke="#f08c00"
        d="M113.754 100.817c11.4.17 59.23 2.5 70.4 2.69m-74.08 2.48c11.02.6 59.89 1.11 72.65.76"
      />
      <path
        fill="none"
        stroke="#f08c00"
        d="M157.954 116.487c9.17-5.46 16.45-7.83 26.21-9.95m-25.59 9.12c5.71-1.81 11.59-4.93 24.48-9.62"
      />
      <path
        fill="none"
        stroke="#f08c00"
        d="M157.864 99.387c9.19.64 16.5 4.37 26.3 7.15m-25.69-7.98c5.58 2.05 11.48 2.79 24.58 7.48"
      />
    </g>
    <path
      fill="none"
      stroke="#1e1e1e"
      className="animate-pulse"
      strokeWidth={0.5}
      d="M253.991 93.883c4.5-1.65 3.55-4.18 5.13-7.93m-4.36 7.6c1.37-3.16 2.28-3.84 6.68-8.36m-14.33 22.26c5-7.39 15.21-12.81 24.37-23.28m-22.75 23.35c5.26-7.59 6.01-11.77 19.18-25.67m-19.08 31.31c6.04-12.97 13.12-20.12 21.33-27.11m-19.12 21.82c6.77-5.91 14.93-13.78 21.46-24.5m4.6-5.78c1.34-1.57 5.01-3.08 5.71-6.29m-5.44 7.87c2.46-1.88 4.08-4.09 6.03-7.26m-29.66 42.53c5.65-12.92 18.91-23.91 25.9-28.53m-27.97 27.2c9.48-11.56 19.51-20.63 27.3-27.41m2.18 1.78c-2.52-4.12 3.75-7.26 7.01-15.72m-11.61 14.17c3.29-3.66 6.27-9.81 11.35-13.36m-31.23 41.06c9.59-15.23 24.47-25.15 40.52-39.93m-43.38 43.68c13.13-14.68 26.39-28.13 39.77-44.06m-35.62 45.69c6.15-12.79 17.07-19.09 42.23-41.37m-40.9 39.25c13.82-12.23 23.85-25.24 38.31-41.52m-38.46 42.32c7.95-7.83 19.28-20.59 39.65-41.32m-35.16 42.62c10.42-13.32 26.42-26.92 35.68-43.34m-30.94 49.4c10.55-20.93 27.37-32.45 41.54-58.02m-42.34 53.67c16.78-18.85 31.79-36.86 43.61-49.56m-42.64 51.58c22.47-19.72 40.12-45.04 50.33-56.57m-49.54 58.06c11.99-16.84 27.29-29.97 50.4-55.95m-45.11 56.58c11.1-10.09 20.85-23.71 50.44-61.41m-51.21 61.85c11.95-14.46 25.86-26.22 53.21-58.49m-45.95 64.26c10.74-14.31 21.27-28.64 48.99-67.98m-53.96 65.27c13.47-14.64 26.47-30.9 54.73-65.03m-45.36 67.7c12.67-25.63 31.34-48.81 49.83-64.46m-54.78 62.6c19.82-20.95 40.12-46.32 57.32-65.05m-49.56 68.04c19.1-23.34 40.26-50.35 52.27-70.28m-55.85 66.77c18.24-18.4 39.72-41.46 57.65-64.14m-51.23 69.12c21.63-31.85 41.96-49.52 60.38-73.63m-61.96 69.06c13.91-10.9 25.44-25.61 59.99-65.55m-54.71 68.86c16.51-17.99 27.49-32.89 58.29-70.67m-59.95 72.37c24.86-29.7 47.23-54 59.6-69.35m-51.96 68.58c14.42-15.31 30.89-40.81 59.57-67.08m-60.23 66.93c17.56-19.52 36.37-41.47 56.02-66.02m-50.28 68.01c14.56-18.02 23.59-31.59 59.17-70.07m-60.81 68.4c19.52-23.19 41.61-51.78 57.06-64.41m-57.65 63.51c25.98-19.67 42-40.86 62.83-61.27m-60.87 64.96c16.56-20.47 33.68-37.34 60.19-64.74m-50.58 66.04c5.15-19.61 19.63-27.16 51.02-65.4m-52.79 64.02c13.41-14.09 24.15-25.92 52.49-63.1m-51.27 64.99c9.62-16.73 24.41-24.67 54.32-56.86m-49.68 56.97c12.59-17.87 28.71-35.32 50.01-62.21m-46.2 60.12c16.22-16.05 26.13-31.82 42.12-48.55m-42.21 50.14c7.97-10.02 15.22-19.65 38.49-48.13m2.94-1.12c-.37-.46 1.67-3.4 2.53-2.83m-3.97 3.9c1.47-1 2.12-1.77 4.91-5.09m-38.47 52.05c9.81-15.57 20.62-22.73 40.77-48.88m-39.26 45.99c11.64-14.49 25.66-29.92 37.74-41.73m-32.91 46.08c11.27-17.46 24.71-37.18 36.15-44.83m-37.66 40.07c10.51-9.71 20.79-23.84 40.11-43.4m-32.89 43.51c.44-.3.61-.88.89-.82m-.82.71c.28-.27.56-.62.67-.83m1.73-4.64c12.58-9.75 23.23-23.68 30.68-35.6m-29.38 39.89c9.94-13.69 20.55-22.68 33.27-39.2m-36.41 44.4c19.94-19.27 31.21-36.99 43.24-41.09m-41.26 36.32c11.24-9.39 21.3-20.11 37.8-39.94m-29.64 40.98c6.91-8.26 20.24-19.97 36.38-40.08m-36.92 38.64c13.61-13.62 23.38-27.24 32.36-37.44m-25.92 36.75c8.49-4.54 13.36-15.45 33.34-36.54m-35.29 36.15c10.72-7.82 15.01-14.84 31.58-31.84m-25.81 36.23c12.18-9.87 18.21-18.57 27.49-32.15m-25.26 27.41c9.1-9.47 17.04-19.06 27.02-29.85m-22.15 35.03c4.32-10.84 9.05-20.33 21.67-33.57m-21.22 29.64c4.9-6.21 13.41-14.28 22.65-28.06m-18.11 22.42c4.78-.95 8.06-6.6 23.45-16.23m-18.36 16.63c.28-1.74 7.63-8.03 14.01-16.69"
    />
    <path
      fill="#1e1e1e"
      className="animate-pulse"
      d="m253.391 96.473-.27-.96q-.28-.95-.47-2.23-.19-1.28-.22-2.47-.04-1.19.5-2.55.54-1.36 1.39-2 .85-.64 2.03-1.23 1.19-.59 2.28-.99 1.1-.39 2.28-.65 1.19-.26 3.43-.48 2.24-.22 3.83-.29 1.58-.06 3.05-.08 1.47-.01 2.8.15 1.32.16 2.42.64 1.11.48-.07.03-1.17-.45-1.33-1.88-.15-1.44-.02-3.09.13-1.66 1.37-3.28 1.25-1.62 2.82-2.52 1.56-.9 3.45-1.29 1.88-.4 3.01-.52t3.33-.2q2.2-.08 4.14-.03 1.93.05 3.42.47 1.5.42 2.64 1.12 1.15.7 2.15 1.85.99 1.15 1.46 1.86.47.71.55.85.08.13.13.29.04.15.05.3.01.16-.01.32-.03.15-.1.3-.06.14-.16.27-.1.12-.22.22-.13.1-.27.16-.14.07-.3.1-.16.03-.31.02-.16-.01-.31-.06-.15-.05-.29-.13-.14-.08-.25-.19-.11-.11.2-1.49.3-1.37 1.2-2.37.91-1 2-1.91 1.1-.92 2.23-1.82 1.14-.91 2.36-1.69 1.23-.78 3.02-1.67 1.8-.89 4-1.59 2.2-.7 4.67-1.23 2.47-.54 5.31-1.01 2.84-.46 5.73-.72 2.89-.26 5.65-.38 2.75-.11 5.71-.15 2.96-.05 5.97.42 3.01.46 5.81 1.15 2.8.68 5.16 1.66 2.36.97 4.13 2.04 1.76 1.08 3.08 3.2 1.32 2.13 1.86 4.19.53 2.06.75 3.97.22 1.91.02 3.51-.2 1.6-.9 2.92-.71 1.31-2.05 2.83-1.34 1.53-2.91 2.25-1.57.71-2.68.91-1.1.2-1.4.22-.3.02-.46.02-.16 0-.31-.04-.16-.03-.3-.11-.14-.07-.26-.17-.11-.11-.21-.24-.09-.13-.14-.28-.06-.14-.08-.3-.02-.16 0-.32.02-.15.07-.3.06-.15.15-.28.09-.13.21-.24.11-.1.25-.18.14-.07.3-.11.15-.04 1.47-.12 1.32-.07 3.65-.08 2.32-.01 4.27-.01 1.96-.01 6.11 1.09 4.16 1.09 6.63 2.12 2.47 1.03 4.77 3.07 2.3 2.04 3.87 4.75 1.57 2.71 2.25 6.22.67 3.5-1.31 7.9-1.99 4.4-4.99 7.61t-6.69 5.5q-3.69 2.29-7.65 3.82-3.96 1.52-7.89 2.35-3.94.83-6.72 1.18t-5.36.5q-2.58.15-4.68.1-2.09-.06-3.64-.4-1.54-.34-3.17-1.14-1.63-.8-2.01-1.3-.39-.5-.49-.62-.09-.13-.16-.27-.06-.15-.09-.3-.03-.16-.02-.32.01-.16.06-.31.05-.15.13-.28.08-.14.19-.25.12-.11.25-.19.14-.08.29-.13.15-.05.31-.06.16-.01.31.02.16.03.3.1.15.06.27.16.13.1 1.22-.24 1.09-.34 2.34-.05 1.25.28 1.35.47.11.19.25.27.14.07.26.18.11.1.21.23.09.13.14.28.06.15.08.3.02.16 0 .32-.02.15-.07.3-.06.15-.15.28-.09.13-.21.24-.11.1-.25.18-.14.07-.3.11-.15.04-.31.04t-.31-.04q-.16-.04-3.04 1.19t-6.06 1.91q-3.17.67-6.76 1.08-3.58.41-6.47.64-2.89.23-5.67.27-2.78.04-5.24-.19-2.47-.22-5.04-.62t-5.2-.99q-2.63-.6-5.26-1.27t-5.71-1.69l-6.01-1.97q-2.94-.97-5.58-1.92-2.65-.95-4.88-1.84-2.23-.89-4.45-1.94-2.22-1.05-4.27-2.16-2.06-1.11-3.64-2.07-1.57-.96-3.46-2.08-1.89-1.11-2.86-1.79-.98-.68-2.14-1.66-1.15-.99-2.3-2.16-1.15-1.16-1.9-2.26-.75-1.09-1.22-2.88-.46-1.78-.18-3.42.29-1.64.66-2.9.37-1.26 1.07-2.12.7-.85 1.75-1.98 1.04-1.13 2.56-2 1.52-.87 1.73-.91.21-.05.42-.02.21.03.4.12.18.09.33.24.15.15.25.34.1.19.13.4.03.21-.01.42t-.14.4q-.11.18-.26.32-.16.15-.34.23-.2.09-.41.11-.21.02-.42-.03t-.38-.17q-.18-.11-.32-.27-.13-.17-.21-.37-.07-.2-.08-.41-.01-.21.05-.41.06-.21.19-.38.12-.17.29-.3.17-.12.37-.19.21-.06.42-.06.21.01.41.08.2.07.36.2.17.13.28.31.12.18.17.38.05.21.04.42-.02.21-.1.41-.08.2-.22.36-.14.15-.32.26l-.18.11-1.1.61q-1.1.61-1.94 1.6-.84.98-1.61 1.84-.77.86-1.06 2.17-.28 1.3-.29 2.55-.01 1.25.74 2.7.75 1.45 1.84 2.57 1.09 1.11 2.14 2.01 1.05.9 1.94 1.53.9.63 2.8 1.75 1.9 1.13 3.42 2.05 1.52.93 3.52 2.01 1.99 1.08 4.13 2.1 2.14 1.01 4.33 1.88 2.2.88 4.8 1.82 2.61.94 5.55 1.9 2.94.97 5.93 1.95 2.99.99 5.58 1.66 2.6.66 5.14 1.24 2.54.57 5.03.96t4.82.61q2.33.22 5.02.19 2.69-.04 5.54-.27 2.85-.22 6.3-.61 3.46-.39 6.51-1.03 3.04-.64 5.12-1.31t3.75-1.21q1.67-.54 1.81-.46.14.07.26.18.11.1.21.23.09.13.14.28.06.15.08.3.02.16 0 .32-.02.15-.07.3-.06.15-.15.28-.09.13-.21.24-.11.1-.25.18-.14.07-.3.11-.15.04-.31.04t-.31-.04q-.16-.04-.53-.38-.38-.34-1.53-.38-1.15-.03-2.54-.1-1.39-.06-1.49-.18-.09-.13-.16-.27-.06-.15-.09-.3-.03-.16-.02-.32.01-.16.06-.31.05-.15.13-.28.08-.14.19-.25.12-.11.25-.19.14-.08.29-.13.15-.05.31-.06.16-.01.31.02.16.03.3.1.15.06.27.16.13.1.36.47.23.36 1.37.94 1.15.58 2.44.89 1.29.3 3.27.36t4.47-.08q2.49-.14 5.17-.47 2.67-.34 6.41-1.11 3.74-.78 7.48-2.19 3.74-1.42 7.17-3.5 3.44-2.09 6.21-4.96 2.77-2.88 4.65-6.61 1.89-3.73 1.34-6.7-.56-2.96-1.88-5.35-1.32-2.38-3.24-4.17-1.93-1.8-4.22-2.76-2.29-.96-6.11-2.01-3.82-1.05-5.76-1.05t-4.24.01q-2.31.01-3.49.08-1.18.07-1.34.07-.16 0-.31-.04-.16-.03-.3-.11-.14-.07-.26-.17-.11-.11-.21-.24-.09-.13-.14-.28-.06-.14-.08-.3-.02-.16 0-.32.02-.15.07-.3.06-.15.15-.28.09-.13.21-.24.11-.1.25-.18.14-.07.3-.11.15-.04.79-.1.63-.06 1.74-.38 1.11-.32 2.19-.98 1.08-.65 2.15-2.29 1.07-1.65 1.26-2.9.18-1.26-.01-2.99-.18-1.73-.59-3.46-.41-1.72-1.32-3.43-.91-1.71-2.44-2.64-1.54-.92-3.71-1.84-2.17-.91-4.86-1.57-2.69-.66-5.48-1.11-2.79-.45-5.71-.41-2.93.05-5.61.16-2.69.1-5.49.35-2.79.25-5.57.7-2.78.46-5.13.96-2.35.51-4.37 1.13-2.02.63-3.69 1.44-1.67.82-2.77 1.52-1.11.7-2.23 1.6-1.12.89-2.09 1.68-.97.8-2.11 1.46-1.13.67-1.05.81.08.13.13.29.04.15.05.3.01.16-.01.32-.03.15-.1.3-.06.14-.16.27-.1.12-.22.22-.13.1-.27.16-.14.07-.3.1-.16.03-.31.02-.16-.01-.31-.06-.15-.05-.29-.13-.14-.08-.25-.19-.11-.11-.57-.82-.46-.7-1.12-1.56-.67-.85-1.56-1.43-.9-.58-2.04-.93-1.14-.34-2.96-.39-1.82-.04-3.9.03t-4.08.31q-1.99.25-3.23.68-1.23.43-2.42 1.26-1.2.83-1.57 1.99-.38 1.15-.34 2.38.04 1.23.29 2.41.25 1.18.03 2.52t-1.48 1.11l-2.46-.47q-1.21-.23-2.65-.22-1.44.01-2.95.07-1.51.07-3.62.27-2.12.21-3.6.56-1.48.35-2.55.78-1.07.43-2.31 1.22-1.23.78-1.21 1.86.02 1.08.18 2.49.16 1.41.43 2.37.28.95.3 1.11.03.16.01.31-.01.16-.06.31-.05.15-.14.29-.08.13-.19.24-.12.11-.25.19t-.29.12q-.15.04-.31.05-.15.01-.31-.03-.15-.03-.3-.1-.14-.06-.26-.16-.12-.11-.22-.23-.09-.13-.16-.27l-.06-.15Z"
    />
    <g strokeLinecap="round">
      <path
        fill="none"
        stroke="#1e1e1e"
        d="M217.31 104.735c5.09.02 25.26-.1 30.37-.19m-31.48-.32c4.96.14 25.66 1.57 30.91 1.75"
      />
      <path
        fill="none"
        stroke="#1e1e1e"
        d="M233.89 111.305c3.31-1.24 7.82-3.53 13.39-5.56m-14.18 4.88c3.62-.9 6.74-2.41 13.79-4.52"
      />
      <path
        fill="none"
        stroke="#1e1e1e"
        d="M234.44 100.945c3.19 2.26 7.51 3.47 12.84 4.8m-13.63-5.48c3.58 1.76 6.56 2.91 13.24 5.84"
      />
    </g>
    <text
      y={14.016}
      fill="#e03131"
      dominantBaseline="alphabetic"
      fontFamily="Virgil, Segoe UI Emoji"
      fontSize={16}
      style={{
        whiteSpace: "pre",
      }}
      transform="translate(295.049 90.124)"
    >
      {"Server"}
    </text>
    <path
      fill="none"
      fillOpacity={0.3}
      stroke="#1e1e1e"
      strokeDasharray="1.5 7"
      strokeLinecap="round"
      strokeOpacity={0.3}
      strokeWidth={1.5}
      d="M153.164 10.003c-.31 31.82-.21 157.33.24 189.12"
    />
    <text
      y={14.016}
      fill="#1e1e1e"
      dominantBaseline="alphabetic"
      fontFamily="Virgil, Segoe UI Emoji"
      fontSize={16}
      style={{
        whiteSpace: "pre",
      }}
      transform="translate(119.411 70.26)"
    >
      {"Stream"}
    </text>
  </svg>
);
export default Client;
