import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import axios from "axios";
import FormData from "form-data";
import { useState } from "react";
import BackgroundPattern from "@/components/ui/background-pattern";
import { cn } from "@/lib/utils";
const classes = [
  "Safe Driving",
  "Texting right",
  "Talking on the Phone right",
  "Texting left",
  "Talking on the phone left",
  "Opearting the radio",
  "Drinking",
  "Reaching Behind",
  "Hair & Makeup",
  "Talking to passenger",
];
function Model() {
  const [img, setImg] = useState(null);
  const [sta, setSta] = useState(null);
  const [prev, setPrev] = useState(null);
  const onChange = (e) => {
    setImg(e.target.files[0]);
    const reader = new FileReader();
    reader.onload = () => {
      setPrev(reader.result);
    };
    reader.readAsDataURL(e.target.files[0]);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("image", img);
    const res = await axios.post("http://localhost:3000/classify", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    setSta(res.data.class);
  };

  const isActive = (i) => {
    if (sta && sta === i) return "bg-black text-gray-200";
  };
  return (
    <div className="relative flex space-x-20 bg-gray-200 px-20 2xl:px-40 py-10 w-full h-screen">
      <BackgroundPattern
        background="[&>*:nth-child(odd)]:bg-black/5"
        border="border-black/5 hover:border-black/20"
      />
      <div className="z-10 space-y-6 w-1/2 h-full">
        <form className="z-10 space-y-8 w-full max-w-fit">
          <Label
            htmlFor="picture"
            className="font-extrabold font-libre text-3xl text-black"
          >
            Upload Picture
          </Label>
          <span className="flex gap-8">
            <Input
              id="picture"
              type="file"
              className="space-x-2 file:border-0 file:bg-Lora hover:file:hover:bg-Dora file:mr-4 file:px-4 placeholder:px-2 file:py-[0.61rem] p-0 border-none file:font-semibold file:text-sm file:text-white overflow-clip"
              accept=".png, .jpg, .jpeg, .jfif, .pjpeg, .pjp, .webp"
              onChange={onChange}
            />
            <Button
              className="bg-Lora hover:bg-Dora font-libre"
              onClick={onSubmit}
            >
              Submit
            </Button>
          </span>
        </form>
        {prev ? (
          <div className="bg-Lora shadow-2xl rounded-2xl w-fit">
            {/* <AspectRatio ratio={16 / 9} className="z-10 bg-muted"> */}
            <img className="rounded-2xl object-contain" src={prev} />
            {/* </AspectRatio> */}
          </div>
        ) : (
          <div className="bg-Lora shadow-2xl rounded-2xl">
            <AspectRatio ratio={16 / 9} className="z-10 bg-muted"></AspectRatio>
          </div>
        )}
      </div>
      <div className="z-10 mx-auto w-1/2 h-full font-libre text-black text-center">
        <ul className="place-content-start gap-2 grid grid-cols-5 mt-36 h-full font-bold text-frank text-sm">
          {classes.map((title, i) => (
            <li
              key={i}
              className={cn(
                "bg-white hover:bg-black/50 h-28 rounded-lg w-full hover:text-gray-200 transition flex items-center justify-center",
                isActive(i)
              )}
            >
              <p>{title}</p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default Model;
