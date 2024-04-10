import React from "react";
export default function SubHeading({ children, id }) {
  return (
    <div id={id} className="pt-6">
      <span className="space-y-4">
        <h2 className="font-bold font-libre text-Lora/80 text-xl">
          {children}
        </h2>
      </span>
    </div>
  );
}
