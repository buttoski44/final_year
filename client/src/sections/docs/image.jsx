import { AspectRatio } from "@radix-ui/react-aspect-ratio";
export default function Image({ link, children }) {
  return (
    <div className="-z-50 h-full">
      <AspectRatio ratio={16 / 9}>
        {link ? <img src={link} alt="" /> : children}
      </AspectRatio>
    </div>
  );
}
