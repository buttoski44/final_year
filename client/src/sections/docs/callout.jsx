import { Alert, AlertLogo, AlertTitle } from "@/components/ui/alert";
import Bulb from "@/assets/bulb";
import Underline from "./underline";
export default function Callout({ logo, children, className }) {
  return (
    <Alert className={className}>
      <AlertLogo>
        <Bulb />
      </AlertLogo>
      <AlertTitle>
        <Underline className="m-0 p-0 decoration-yellow-400 text-black">
          {children}
        </Underline>
      </AlertTitle>
    </Alert>
  );
}
