import { Separator } from "@/components/ui/separator";
import React from "react";
export default function Heading({ children, id }) {
  return (
    <div id={id}>
      <span className="space-y-4">
        <h6 className="font-bold font-libre text-3xl text-Lora">{children}</h6>
        <Separator className="bg-Lora" />
      </span>
    </div>
  );
}
