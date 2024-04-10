import React from "react";
export default function CodeBlock(props) {
  const { fileName, code } = props.file;
  return (
    <div className="py-2">
      <div className="border-gray-200 bg-black/30 border rounded-md w-4/5 font-semibold text-gray-200">
        <p className="px-8 py-2 border">
          <code>{fileName}</code>
        </p>
        <pre className="px-8 pb-4 font-thin text-md leading-8">
          <code className="">{code}</code>
        </pre>
      </div>
    </div>
  );
}
