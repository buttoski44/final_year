import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
export default function Indexes({ index }) {
  return (
    <li>
      <Accordion type="single" collapsible>
        <AccordionItem value="item-1">
          <AccordionTrigger className="">{index.Title}</AccordionTrigger>
          <AccordionContent>
            <ul className="px-6">
              {index.List.map((list, i) => (
                <li
                  key={i}
                  className="flex justify-end border-gray-400 hover:border-white focus:border-white px-4 py-2 border-r-2"
                >
                  <a href={`#${list.id}`} className="scroll-smooth">
                    {list.title}
                  </a>
                </li>
              ))}
            </ul>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </li>
  );
}
