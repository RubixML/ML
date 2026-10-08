<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\AllowMockObjectsWithoutExpectations;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Transformers\ImageFlipper;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

#[AllowMockObjectsWithoutExpectations]
#[Group('Transformers')]
#[RequiresPhpExtension('gd')]
#[CoversClass(ImageFlipper::class)]
class ImageFlipperTest extends TestCase
{
    protected ImageFlipper $transformer;

    public static function invalidProbabilityProvider() : array
    {
        return [
            'negative horizontal' => [-0.1, 0.0],
            'horizontal greater than one' => [1.1, 0.0],
            'negative vertical' => [0.0, -0.1],
            'vertical greater than one' => [0.0, 1.1],
        ];
    }

    protected function setUp() : void
    {
        $this->transformer = new ImageFlipper(horizontal: 0.5, vertical: 0.0);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(ImageFlipper::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    #[Test]
    #[DataProvider('invalidProbabilityProvider')]
    public function transformWithInvalidProbability(float $horizontal, float $vertical) : void
    {
        $this->expectException(InvalidArgumentException::class);

        new ImageFlipper($horizontal, $vertical);
    }

    #[Test]
    public function transformHorizontally() : void
    {
        $source = $this->stripedImage();

        $expected = $this->copyOf($source);

        imageflip($expected, IMG_FLIP_HORIZONTAL);

        $mock = $this->partialMock(0.5, 0.0, fn (float $probability) : bool => $probability === 0.5);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSameImage($expected, $sample[0]);
        $this->assertSame(2, imagesx($sample[0]));
        $this->assertSame(2, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);
    }

    #[Test]
    public function transformVertically() : void
    {
        $source = $this->stripedImage();

        $expected = $this->copyOf($source);

        imageflip($expected, IMG_FLIP_VERTICAL);

        $mock = $this->partialMock(0.5, 1.0, fn (float $probability) : bool => $probability === 1.0);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSameImage($expected, $sample[0]);
        $this->assertSame(2, imagesx($sample[0]));
        $this->assertSame(2, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
    }

    #[Test]
    public function transformBothAxes() : void
    {
        $source = $this->stripedImage();

        $expected = $this->copyOf($source);

        imageflip($expected, IMG_FLIP_HORIZONTAL);
        imageflip($expected, IMG_FLIP_VERTICAL);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $dataset->apply(new ImageFlipper(horizontal: 1.0, vertical: 1.0));

        $sample = $dataset->sample(0);

        $this->assertSameImage($expected, $sample[0]);
        $this->assertSame(2, imagesx($sample[0]));
        $this->assertSame(2, imagesy($sample[0]));
    }

    #[Test]
    public function transformWithoutFlipping() : void
    {
        $source = $this->stripedImage();

        $expected = $this->copyOf($source);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $dataset->apply(new ImageFlipper(horizontal: 0.0, vertical: 0.0));

        $sample = $dataset->sample(0);

        $this->assertSameImage($expected, $sample[0]);
        $this->assertSame(2, imagesx($sample[0]));
        $this->assertSame(2, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
    }

    #[Test]
    public function transformAlwaysFlips() : void
    {
        $samples = [];

        for ($i = 0; $i < 100; ++$i) {
            $samples[] = [$this->stripedImage()];
        }

        $dataset = Unlabeled::quick($samples);

        $dataset->apply(new ImageFlipper(horizontal: 1.0, vertical: 0.0));

        [$topLeft, $topRight] = $this->stripedColors();

        foreach ($dataset->samples() as [$sample]) {
            $this->assertSame($topRight, imagecolorat($sample, 0, 0));
            $this->assertSame($topLeft, imagecolorat($sample, 1, 0));
        }
    }

    #[Test]
    public function transformNeverFlips() : void
    {
        $samples = [];

        for ($i = 0; $i < 100; ++$i) {
            $samples[] = [$this->stripedImage()];
        }

        $dataset = Unlabeled::quick($samples);

        $dataset->apply(new ImageFlipper(horizontal: 0.0, vertical: 0.0));

        [$topLeft, $topRight] = $this->stripedColors();

        foreach ($dataset->samples() as [$sample]) {
            $this->assertSame($topLeft, imagecolorat($sample, 0, 0));
            $this->assertSame($topRight, imagecolorat($sample, 1, 0));
        }
    }

    #[Test]
    public function transformApproximatelyRespectsProbability() : void
    {
        $n = 1000;

        $samples = [];

        for ($i = 0; $i < $n; ++$i) {
            $samples[] = [$this->stripedImage()];
        }

        $dataset = Unlabeled::quick($samples);

        $dataset->apply(new ImageFlipper(horizontal: 0.5, vertical: 0.0));

        [, $topRight] = $this->stripedColors();

        $flipped = 0;

        foreach ($dataset->samples() as [$sample]) {
            if (imagecolorat($sample, 0, 0) === $topRight) {
                ++$flipped;
            }
        }

        $this->assertGreaterThan(0.4 * $n, $flipped);
        $this->assertLessThan(0.6 * $n, $flipped);
    }

    #[Test]
    public function transformHonorsVerticalProbability() : void
    {
        $n = 1000;

        $samples = [];

        for ($i = 0; $i < $n; ++$i) {
            $samples[] = [$this->stripedImage()];
        }

        $dataset = Unlabeled::quick($samples);

        $dataset->apply(new ImageFlipper(horizontal: 0.0, vertical: 0.5));

        [, , $bottomLeft] = $this->stripedColors();

        $flipped = 0;

        foreach ($dataset->samples() as [$sample]) {
            if (imagecolorat($sample, 0, 0) === $bottomLeft) {
                ++$flipped;
            }
        }

        $this->assertGreaterThan(0.4 * $n, $flipped);
        $this->assertLessThan(0.6 * $n, $flipped);
    }

    #[Test]
    public function transformExactlyOnce() : void
    {
        $dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever', 69],
        ]);

        $dataset->apply($this->transformer);

        $sample = $dataset->sample(0);

        self::assertTrue(is_resource($sample[0]) || $sample[0] instanceof \GdImage);
        self::assertEquals(32, imagesx($sample[0]));
        self::assertEquals(32, imagesy($sample[0]));
        self::assertSame('whatever', $sample[1]);
        self::assertEquals(69, $sample[2]);
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $this->assertSame('Image Flipper (horizontal: 0.5, vertical: 0)', (string) $this->transformer);
    }

    /**
     * Build an Image Flipper whose coin flips are driven by the given callback.
     *
     * @param float $horizontal
     * @param float $vertical
     * @param callable(float) : bool $shouldFlip
     * @return ImageFlipper
     */
    protected function partialMock(float $horizontal, float $vertical, callable $shouldFlip) : ImageFlipper
    {
        $mock = $this->getMockBuilder(ImageFlipper::class)
            ->onlyMethods(['shouldFlip'])
            ->enableOriginalConstructor()
            ->setConstructorArgs([$horizontal, $vertical])
            ->getMock();

        $mock->method('shouldFlip')->willReturnCallback($shouldFlip);

        return $mock;
    }

    /**
     * Build a 2x2 truecolor image whose pixels all differ so that any permutation is detectable.
     *
     * @return \GdImage
     */
    protected function stripedImage() : \GdImage
    {
        $image = imagecreatetruecolor(2, 2);

        imagesetpixel($image, 0, 0, imagecolorallocate($image, 255, 0, 0));
        imagesetpixel($image, 1, 0, imagecolorallocate($image, 0, 255, 0));
        imagesetpixel($image, 0, 1, imagecolorallocate($image, 0, 0, 255));
        imagesetpixel($image, 1, 1, imagecolorallocate($image, 255, 255, 0));

        return $image;
    }

    /**
     * Return the colors of the four corners of a striped image in reading order.
     *
     * @return array{int, int, int, int}
     */
    protected function stripedColors() : array
    {
        $image = $this->stripedImage();

        return [
            imagecolorat($image, 0, 0),
            imagecolorat($image, 1, 0),
            imagecolorat($image, 0, 1),
            imagecolorat($image, 1, 1),
        ];
    }

    /**
     * Copy an image into a new truecolor image of the same dimensions.
     *
     * @param \GdImage $image
     * @return \GdImage
     */
    protected function copyOf(\GdImage $image) : \GdImage
    {
        $copy = imagecreatetruecolor(imagesx($image), imagesy($image));

        imagecopy($copy, $image, 0, 0, 0, 0, imagesx($image), imagesy($image));

        return $copy;
    }

    /**
     * Assert that two images contain the same pixels.
     *
     * @param \GdImage $expected
     * @param \GdImage $actual
     */
    protected function assertSameImage(\GdImage $expected, mixed $actual) : void
    {
        $this->assertInstanceOf(\GdImage::class, $actual);

        $this->assertSame(imagesx($expected), imagesx($actual));
        $this->assertSame(imagesy($expected), imagesy($actual));

        for ($x = 0; $x < imagesx($expected); ++$x) {
            for ($y = 0; $y < imagesy($expected); ++$y) {
                $this->assertSame(
                    imagecolorat($expected, $x, $y),
                    imagecolorat($actual, $x, $y),
                    "Pixel ($x, $y) does not match."
                );
            }
        }
    }
}
