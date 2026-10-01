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
use Rubix\ML\Transformers\ImageRotator;
use Rubix\ML\Transformers\Transformer;
use PHPUnit\Framework\TestCase;

#[AllowMockObjectsWithoutExpectations]
#[Group('Transformers')]
#[RequiresPhpExtension('gd')]
#[CoversClass(ImageRotator::class)]
class ImageRotatorTest extends TestCase
{
    protected ImageRotator $transformer;

    public static function invalidParameterProvider() : array
    {
        return [
            'offset below range' => [-360.1, 0.2],
            'offset above range' => [360.1, 0.2],
            'negative jitter' => [0.0, -0.1],
            'jitter above range' => [0.0, 1.1],
        ];
    }

    public static function rotationProvider() : array
    {
        return [
            '90 degrees' => [90.0, ['red' => 0, 'green' => 0, 'blue' => 255, 'alpha' => 0]],
            '270 degrees' => [270.0, ['red' => 255, 'green' => 0, 'blue' => 0, 'alpha' => 0]],
        ];
    }

    public static function degreesProvider() : array
    {
        return [
            '90 degrees' => [90.0],
            '270 degrees' => [270.0],
        ];
    }

    public static function invalidFillColorProvider() : array
    {
        return [
            'empty string' => [''],
            'missing hash and too short' => ['fff'],
            'too long' => ['#ffffffff'],
            'not hex' => ['#gggggg'],
            'named color' => ['white'],
            'rgb literal' => ['255,255,255'],
        ];
    }

    public static function fillColorProvider() : array
    {
        return [
            'black with hash' => ['#000000', 0, 0, 0],
            'black without hash' => ['000000', 0, 0, 0],
            'white' => ['#ffffff', 255, 255, 255],
            'green' => ['#00ff00', 0, 255, 0],
            'uppercase' => ['#FF00FF', 255, 0, 255],
        ];
    }

    protected function setUp() : void
    {
        $this->transformer = new ImageRotator(offset: 0.0, jitter: 0.2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(ImageRotator::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
    }

    #[Test]
    #[DataProvider('invalidParameterProvider')]
    public function invalidParameter(float $offset, float $jitter) : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new ImageRotator($offset, $jitter);
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertContainsOnlyInstancesOf(\Rubix\ML\DataType::class, $this->transformer->compatibility());
    }

    #[Test]
    public function transformWithDefaultJitter() : void
    {
        $transformer = new ImageRotator(0.0);

        $dataset = Unlabeled::quick([
            [imagecreatefrompng('./tests/test.png'), 'whatever', 69],
        ]);

        $dataset->apply($transformer);

        $sample = $dataset->sample(0);

        $image = $sample[0];

        $this->assertEquals(32, imagesx($image));
        $this->assertEquals(32, imagesy($image));
        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);
    }

    #[Test]
    public function transformDoesNotRotateBackToOriginal() : void
    {
        $source = imagecreatetruecolor(32, 32);

        imagefilledrectangle($source, 0, 0, 15, 31, imagecolorallocate($source, 255, 0, 0));
        imagefilledrectangle($source, 16, 0, 31, 31, imagecolorallocate($source, 0, 0, 255));

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator(90.0);

        $dataset->apply($mock);

        $image = $dataset->sample(0)[0];

        $this->assertNotSame($source, $image, 'Transformer must return a new image.');
    }

    #[Test]
    public function transformPreservesNonImageColumns() : void
    {
        $dataset = Unlabeled::quick([
            [imagecreatetruecolor(32, 32), 'whatever', 69],
            [imagecreatetruecolor(32, 32), 'nothing', 42],
        ]);

        $mock = $this->mockRotator(90.0);

        $dataset->apply($mock);

        $this->assertSame('whatever', $dataset->sample(0)[1]);
        $this->assertSame(69, $dataset->sample(0)[2]);
        $this->assertSame('nothing', $dataset->sample(1)[1]);
        $this->assertSame(42, $dataset->sample(1)[2]);
    }

    #[Test]
    public function transformWithoutImages() : void
    {
        $dataset = Unlabeled::quick([
            ['whatever', 69],
        ]);

        $mock = $this->mockRotator();

        $mock->expects($this->never())->method('rotationAngle');

        $dataset->apply($mock);

        $this->assertSame('whatever', $dataset->sample(0)[0]);
        $this->assertSame(69, $dataset->sample(0)[1]);
    }

    #[Test]
    public function transformEmptyDataset() : void
    {
        $dataset = Unlabeled::quick([]);

        $dataset->apply($this->transformer);

        $this->assertSame(0, $dataset->numSamples());
    }

    /**
     * The rotated image is resized back to its original dimensions, so a red left half and a
     * blue right half must be swapped to a blue left half and a red right half after 90 degrees.
     */
    #[Test]
    #[DataProvider('rotationProvider')]
    public function transformWideImage(float $degrees, array $expected) : void
    {
        $source = imagecreatetruecolor(80, 40);

        imagefilledrectangle($source, 0, 0, 39, 39, imagecolorallocate($source, 255, 0, 0));
        imagefilledrectangle($source, 40, 0, 79, 39, imagecolorallocate($source, 0, 0, 255));

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator($degrees);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSame(80, imagesx($sample[0]));
        $this->assertSame(40, imagesy($sample[0]));

        $this->assertSame($expected, imagecolorsforindex($sample[0], imagecolorat($sample[0], 5, 5)));

        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);
    }

    #[Test]
    #[DataProvider('degreesProvider')]
    public function transformTallImage(float $degrees) : void
    {
        $source = imagecreatetruecolor(20, 100);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator($degrees);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSame(20, imagesx($sample[0]));
        $this->assertSame(100, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);
    }

    #[Test]
    public function transformExtremeRatioImage90Degrees() : void
    {
        $source = imagecreatetruecolor(200, 5);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator(90.0);

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSame(200, imagesx($sample[0]));
        $this->assertSame(5, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);
    }

    #[Test]
    public function transformSquareImage45Degrees() : void
    {
        $source = imagecreatetruecolor(32, 32);

        imagefilledrectangle($source, 0, 0, 15, 31, imagecolorallocate($source, 255, 0, 0));
        imagefilledrectangle($source, 16, 0, 31, 31, imagecolorallocate($source, 0, 0, 255));

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator();

        $dataset->apply($mock);

        $sample = $dataset->sample(0);

        $this->assertSame(32, imagesx($sample[0]));
        $this->assertSame(32, imagesy($sample[0]));
        $this->assertSame('whatever', $sample[1]);
        $this->assertSame(69, $sample[2]);

        $this->assertNotSame($source, $sample[0], 'Transformer must return a new image.');
    }

    #[Test]
    public function rotationAngleIsWithinRange() : void
    {
        $transformer = new ImageRotator(0.0, 0.2);

        $method = new \ReflectionMethod($transformer, 'rotationAngle');

        for ($i = 0; $i < 1000; ++$i) {
            $angle = $method->invoke($transformer);

            $this->assertGreaterThanOrEqual(0.0, $angle);
            $this->assertLessThan(360.0, $angle);
        }
    }

    #[Test]
    public function zeroJitterReturnsOffset() : void
    {
        $transformer = new ImageRotator(45.0, 0.0);

        $method = new \ReflectionMethod($transformer, 'rotationAngle');

        $this->assertEqualsWithDelta(45.0, $method->invoke($transformer), 0.0001);
    }

    #[Test]
    #[DataProvider('invalidFillColorProvider')]
    public function invalidFillColor(string $fillColor) : void
    {
        $this->expectException(\Rubix\ML\Exceptions\InvalidArgumentException::class);

        new ImageRotator(0.0, 0.2, $fillColor);
    }

    /**
     * A truecolor image must be filled with the requested color. GD reads the background
     * argument to imagerotate() as an RGB value for truecolor images.
     */
    #[Test]
    #[DataProvider('fillColorProvider')]
    public function fillColorOnTruecolorImage(string $fillColor, int $red, int $green, int $blue) : void
    {
        $source = imagecreatetruecolor(32, 32);

        imagefilledrectangle($source, 0, 0, 31, 31, imagecolorallocate($source, 0, 0, 255));

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator(45.0, $fillColor);

        $dataset->apply($mock);

        $image = $dataset->sample(0)[0];

        $colors = imagecolorsforindex($image, imagecolorat($image, 0, 0));

        $this->assertSame($red, $colors['red']);
        $this->assertSame($green, $colors['green']);
        $this->assertSame($blue, $colors['blue']);
    }

    /**
     * A palette image must be filled with the requested color. GD reads the background argument
     * to imagerotate() as a palette index for palette images, so the color has to be allocated
     * against the image rather than passed through as an RGB value.
     */
    #[Test]
    #[DataProvider('fillColorProvider')]
    public function fillColorOnPaletteImage(string $fillColor, int $red, int $green, int $blue) : void
    {
        $source = imagecreate(32, 32);

        imagecolorallocate($source, 255, 255, 255);
        imagecolorallocate($source, 255, 0, 0);

        imagefilledrectangle($source, 0, 0, 31, 31, 1);

        $this->assertFalse(imageistruecolor($source), 'Source must be a palette image.');

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator(45.0, $fillColor);

        $dataset->apply($mock);

        $image = $dataset->sample(0)[0];

        $colors = imagecolorsforindex($image, imagecolorat($image, 0, 0));

        $this->assertSame($red, $colors['red']);
        $this->assertSame($green, $colors['green']);
        $this->assertSame($blue, $colors['blue']);
    }

    /**
     * The default fill color must be black on palette images whose first palette entry is not
     * black. Passing 0 directly would have resolved to whatever color sits at index 0.
     */
    #[Test]
    public function defaultFillColorOnPaletteImageIsBlack() : void
    {
        $source = imagecreate(32, 32);

        imagecolorallocate($source, 255, 255, 255);
        imagecolorallocate($source, 255, 0, 0);

        imagefilledrectangle($source, 0, 0, 31, 31, 1);

        $dataset = Unlabeled::quick([
            [$source, 'whatever', 69],
        ]);

        $mock = $this->mockRotator();

        $dataset->apply($mock);

        $image = $dataset->sample(0)[0];

        $colors = imagecolorsforindex($image, imagecolorat($image, 0, 0));

        $this->assertSame(0, $colors['red']);
        $this->assertSame(0, $colors['green']);
        $this->assertSame(0, $colors['blue']);
    }

    #[Test]
    public function stringRepresentation() : void
    {
        $transformer = new ImageRotator(10.0, 0.5, '#ff0000');

        $this->assertSame('Image Rotator (offset: 10, jitter: 0.5, fillColor: #ff0000)', (string) $transformer);
    }

    /**
     * Build a rotator with a fixed rotation angle. The real constructor must run so that the
     * offset, jitter, and fill color properties are initialized.
     * @param float $degrees
     * @param string $fillColor
     */
    protected function mockRotator(float $degrees = 45.0, string $fillColor = '#000000') : ImageRotator
    {
        $mock = $this->getMockBuilder(ImageRotator::class)
            ->setConstructorArgs([0.0, 0.2, $fillColor])
            ->onlyMethods(['rotationAngle'])
            ->getMock();

        $mock->method('rotationAngle')->willReturn($degrees);

        return $mock;
    }
}
