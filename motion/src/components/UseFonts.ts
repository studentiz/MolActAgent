import { continueRender, delayRender } from "remotion";
import { useEffect, useState } from "react";

export const useFonts = () => {
  const [handle] = useState(() => delayRender("Loading assay typefaces"));

  useEffect(() => {
    Promise.all([
      document.fonts.load('600 32px "DM Sans"'),
      document.fonts.load('400 32px "DM Sans"'),
    ])
      .then(() => continueRender(handle))
      .catch((error) => {
        console.error(error);
        continueRender(handle);
      });
  }, [handle]);
};
