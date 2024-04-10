import * as React from "react";
const Bulb = (props) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    xmlSpace="preserve"
    width={40}
    height={40}
    viewBox="0 0 190.915 190.915"
    {...props}
  >
    <path d="M133.014 71.58c-5.597-20.08-29.743-25.583-47.735-23.08-16.906 2.352-34.789 12.786-37.24 31.377-1.086 8.236-1.071 17.805 4.507 24.637 7.741 9.482 20.967 14.887 22.218 28.174a1.34 1.34 0 0 0-1.219.838c-6.773 1.736-7.582 10.027-2.126 15.35-3.31 3.336-5.345 7.862-1.186 11.729.448.418.974.746 1.483 1.095-8.568 8.541.94 13.448 10.285 15.522.722 6.873 6.175 11.375 12.975 11.734 6.363.337 13.672-4.875 14.33-11.48 2.708-.684 5.145-1.752 6.843-3.561 2.943-3.136 4.126-11.01.94-14.705 4.374-3.311 3.605-8.811-.439-11.791 2.52-1.654 3.602-3.84 1.166-6.646-2.101-2.42-5.392-3.768-8.716-4.549.003-.113.035-.213.02-.332-1.999-14.67 13.09-26.697 20.625-37.658 5.701-8.297 5.877-17.297 3.269-26.654zM95.859 185.604c-5.608.475-9.617-3.164-11.583-7.934 2.672.469 5.213.717 7.158.732 3.722.031 8.986.349 13.911-.208-1.907 3.945-4.111 6.957-9.486 7.41zm-4.425-10.775c-6.888-.059-25.195-1.589-17.505-11.881 5.695 2.617 13.489 2.5 19.044 2.344 6.674-.188 14.701-.949 20.958-4.148 9.907 13.157-15.282 13.745-22.497 13.685zm21.766-17.403c-.018.008-.032.019-.05.027-.353.197-.667.393-1.091.598-5.43 2.629-12.039 3.305-17.967 3.662-5.9.355-12.04.094-17.708-1.688-6.105-1.916-5.481-5.598-2.679-9.573 2.627 1.269 5.834 1.392 8.653 1.581 6.38.427 12.849.098 19.216-.373 1.989-.146 7.447-.959 11.767-2.619 3.226 3.901 4.19 5.93-.141 8.385zm.668-15.15c3.674 2.097-8.319 5.164-9.5 5.289-6.295.676-12.668.851-18.994.746-2.853-.047-5.73-.207-8.53-.773-6.61-1.339-7.488-10.389.265-10.822.393-.023.692-.16.948-.338 7.982 1.301 16.169 1.84 24.181 2.654 4.041.41 8.07 1.212 11.63 3.244zm-20.45-7.889c-1.437-.133-2.873-.275-4.31-.408.005-.022.005-.051.01-.073.049-.155.072-.31.063-.468.773-4.687-1.115-12.223-2.678-16.594-1.679-4.699-4.259-9.111-6.624-13.484-.577-1.065-1.108-2.33-1.648-3.641 3.949-5.264 11.22-6.739 19.272-7.89.282-.04.59-.082.914-.126-4.6 13.548-5.194 28.51-4.999 42.684zm11.936 1.162a42.847 42.847 0 0 0-1.431-.17c-2.395-.242-4.793-.455-7.188-.679.43-13.776.976-27.694 4.938-40.996.114-.383.469-1.348.979-2.657 4.937-.934 10.573-2.84 10.458-7.775-.037-1.611-2.001-1.989-2.865-.776-1.491 2.093-3.563 3.417-5.895 4.372 2.561-6.106 6.197-13.938 7.548-13.66 1.691.349 2.49-1.722 1.138-2.699-5.322-3.841-11.005 12.377-12.191 14.992-.409.904-.731 1.842-1.1 2.762-4.664.989-9.759 1.26-13.867 2.506-3.869 1.174-6.877 3.154-9.027 5.723-.86-1.889-1.829-3.653-3.09-4.845-.793-.75-1.9-.347-2.265.598-.996 2.572 1.13 5.546 2.261 7.806.152.303.323.598.477.901-.988 2.681-1.317 5.737-.858 9.122.141 1.033 1.724.727 1.755-.238.069-2.111.384-3.93.868-5.549 1.858 3.462 3.754 6.904 5.469 10.443 1.843 3.803 2.439 7.217 3.006 11.293.308 2.211.332 5.512 1.536 7.664-2.718-.252-5.436-.5-8.151-.77.093-11.148-7.126-16.83-15.471-23.749-23.893-19.811-5.981-50.532 19.539-56.352 13.531-3.086 29.155-.581 39.956 8.202 16.928 13.766 7.229 32.764-3.368 45.643-6.648 8.085-13.985 17.841-13.161 28.888zM50.55 117.022c-2.363 2.781-4.226 6.025-6.377 8.975-2.068 2.835-5.274 5.266-6.906 8.344-.469.883.313 2.225 1.402 1.823 3.035-1.12 4.883-4.418 6.691-6.944 2.45-3.424 5.073-7.113 6.772-10.977.423-.962-.78-2.164-1.582-1.221zM31.052 84.38c-3.669-2.393-9.155-1.897-13.38-1.886-5.617.015-11.481.595-16.942 1.939-1.115.274-.897 2.106.274 2.023 4.917-.348 9.782-.818 14.719-.841 5.006-.023 10.15 1.641 15.041.979 1.126-.153 1.061-1.71.288-2.214zM55.193 41.729c-1.944-2.811-5.423-4.398-8.223-6.279-3.574-2.4-7.159-4.931-10.976-6.928-1.006-.526-2.218.85-1.266 1.641 3.038 2.526 6.399 4.671 9.626 6.947 3.007 2.123 5.939 5.281 9.454 6.42.988.321 1.998-.916 1.385-1.801zM90.563 3.232c-.24-1.316-1.641-1.649-2.623-.884-1.633 1.273-1.137 4.174-1.141 5.953-.013 7.035-.202 14.421.804 21.397.142.985 1.85.99 1.986 0 .641-4.668.291-9.538.356-14.239.037-2.671.088-5.346.252-8.012.056-.904.307-1.835.316-2.733a7.479 7.479 0 0 0-.021-.782c.076-.225.114-.465.071-.7zM146.323 15.494c-2.682-3.026-6.409 3.279-7.558 4.937-1.742 2.515-3.357 5.118-5.049 7.667-1.279 1.929-3.773 4.146-4.394 6.353-.172.611.09 1.466.71 1.744l.138.062c.633.284 1.336.29 1.856-.239.532-.542.796-1.198 1.535-1.515.566-.244.532-.775.252-1.173.537-.559 1.087-1.106 1.433-1.573 1.317-1.777 2.544-3.623 3.787-5.452 1.207-1.776 2.419-3.549 3.697-5.276a25.475 25.475 0 0 1 1.696-2.043c.327-.356 1.57-1.09.889-1.063 1.278-.051 1.838-1.493 1.008-2.429zM189.113 74.945c-7.936.268-15.862.462-23.788.96-2.688.169-10.849-.489-12.239 2.826-.348.83.903 1.465 1.468.855.166-.179 4.854-.104 5.394-.139 3.306-.212 6.618-.339 9.928-.455 6.394-.223 12.841-.491 19.238-.316 2.45.067 2.352-3.993-.001-3.731zM157.056 121.074c-1.018-1.302-3.066-1.81-4.496-2.542-2.277-1.166-4.504-2.434-6.735-3.684-2.378-1.328-7.142-5.189-10.026-4.248-.538.176-.632.75-.307 1.166 2.073 2.644 6.799 4.414 9.669 6.181 2.764 1.7 7.813 6.161 11.291 5.417 1.113-.239 1.186-1.547.604-2.29z" />
  </svg>
);
export default Bulb;
