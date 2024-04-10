import { cn } from "@/lib/utils";
import React from "react";

export default function Underline({ children, className }) {
  return (
    <p
      className={cn(
        "pt-10 w-full font-biryani text-gray-200 text-sm underline underline-offset-4 leading-56 decoration-2 decoration-Lora decoration-wavy",
        className
      )}
    >
      {children}
    </p>
  );
}
