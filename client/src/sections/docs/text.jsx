import React from "react";

export default function Text({ children }) {
  return (
    <p className="w-full font-biryani text-gray-200 text-sm leading-56">
      {children}
    </p>
  );
}
