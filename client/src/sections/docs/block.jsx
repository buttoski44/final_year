import React from "react";
import { Separator } from "@/components/ui/separator";
import { Terminal } from "lucide-react";
import CodeBlock from "./codeblock";
import { code1 } from "@/codes/codes";
import Callout from "./callout";
import Text from "./text";

export default function Block({ children }) {
  return <div className="space-y-6">{children}</div>;
}
{
  /* <span className="space-y-6">T</span> */
}
{
  /* <Text>
        Lorem ipsum dolor sit amet consectetur adipisicing elit. Ut magni dolor
        odio, impedit illo explicabo aspernatur error saepe odit placeat quis
        iusto fugit animi vero omnis consectetur? Error, ex blanditiis.
      </Text> */
}
{
  /* <CodeBlock file={code1} /> */
}
{
  /* <Callout logo={<Terminal className="w-6 h-6" />} callout="Heads up!" /> */
}
