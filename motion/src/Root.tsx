import { Composition } from "remotion";
import { AssayFilm } from "./AssayFilm";
import { AssayPoster } from "./AssayPoster";
import { AssayPosterMobile } from "./AssayPosterMobile";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="AssayFilm"
        component={AssayFilm}
        durationInFrames={840}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="AssayPoster"
        component={AssayPoster}
        durationInFrames={1}
        fps={30}
        width={1600}
        height={1100}
      />
      <Composition
        id="AssayPosterMobile"
        component={AssayPosterMobile}
        durationInFrames={1}
        fps={30}
        width={780}
        height={2560}
      />
    </>
  );
};
