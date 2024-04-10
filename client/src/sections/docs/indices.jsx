import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import Indexes from "./indexes";

const INDEX_LIST = [
  {
    Title: "Introduction",
    List: [
      {
        id: "background",
        title: "1. Background",
      },
      {
        id: "scope",
        title: "2. Scope",
      },
      {
        id: "architecture",
        title: "3. Architecture",
      },
    ],
  },
  {
    Title: "Serve",
    List: [
      {
        id: "modules",
        title: "1. Importing Required Modules",
      },
      {
        id: "express",
        title:
          "2. Creating an Express App , Loading Pre-Trained Model & Configuring Multer:",
      },
      {
        id: "api",
        title: "3. Handling POST Requests to '/classify' Endpoint:",
      },
    ],
  },
  {
    Title: "Client",
    List: [
      {
        id: "much",
        title: "Not So much !",
      },
    ],
  },
  {
    Title: "Model",
    List: [
      {
        id: "traditional",
        title: "1. Traditional CNN",
      },
      {
        id: "creation",
        title: "2. Model Creation",
      },
      {
        id: "training",
        title: "3. Training",
      },
      {
        id: "transfer",
        title: "4. Image Augmentation for Transfer Learning Models",
      },
      {
        id: "dense",
        title: "5. DenseNet",
      },
      {
        id: "efficient",
        title: "6. EfficientNet",
      },
      {
        id: "mobile",
        title: "7. MobileNet",
      },
      {
        id: "training",
        title: "8. Training Model",
      },
      {
        id: "ensembler",
        title: "9. Ensembler",
      },
    ],
  },
];

export default function Indices() {
  const [filter, setFilter] = useState("");

  const handleChange = (e) => {
    console.log("vsihal");
    setFilter(e.target.value);
  };
  return (
    <>
      <div className="text-right space-y-6 bg-gray-200 rounded-bl-2xl w-1/4 h-full text-black overflow-y-scroll">
        <div className="z-20 absolute flex justify-between bg-black/10 backdrop-blur-sm py-6 pr-6 pl-3 w-1/4 h-20">
          <div className="flex justify-start items-center gap-4">
            <div className="flex">
              <div className="flex justify-center items-center bg-white hover:bg-white shadow-md px-2 rounded-l-md rounded-r-none">
                <Search className="text-gray-200 cursor-text" />
              </div>

              <Input
                type="text"
                className="rounded-l-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-md placeholder:text-gray-300"
                placeholder="Quick Search..."
                onChange={handleChange}
              />
            </div>
          </div>
          <p className="font-extrabold font-libre text-3xl text-black">Index</p>
        </div>
        <div className="relative z-10 flex flex-col justify-end">
          <ul className="px-6 2xl:px-20 pt-20">
            {INDEX_LIST.map((index, i) => {
              const filterIndex = {
                ...index,
                List: index.List.filter(
                  (obj) =>
                    obj.title.toLocaleLowerCase().includes(filter) ||
                    obj.title.includes(filter)
                ),
              };
              return filterIndex.List.length ? (
                <Indexes key={i} index={filterIndex} />
              ) : null;
            })}
          </ul>
        </div>
      </div>
    </>
  );
}
