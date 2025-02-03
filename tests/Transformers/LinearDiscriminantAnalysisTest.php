<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[RequiresPhpExtension('tensor')]
#[CoversClass(LinearDiscriminantAnalysis::class)]
class LinearDiscriminantAnalysisTest extends TestCase
{
    protected Agglomerate $generator;

    protected LinearDiscriminantAnalysis $transformer;

    protected function setUp() : void
    {
        $this->generator = new Agglomerate(
            generators: [
                'red' => new Blob(center: [255, 0, 0], stdDev: 30.0),
                'green' => new Blob(center: [0, 128, 0], stdDev: 10.0),
                'blue' => new Blob(center: [0, 0, 255], stdDev: 20.0),
            ],
            weights: [3, 4, 3]
        );

        $this->transformer = new LinearDiscriminantAnalysis(1);
    }

    public function testFitTransform() : void
    {
        $dataset = $this->generator->generate(30);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $sample = $this->generator->generate(3)
            ->apply($this->transformer)
            ->sample(0);

        $this->assertCount(1, $sample);
    }

    public function testTransformUnfitted() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->generator->generate(1)->samples();

        $this->transformer->transform($samples);
    }
}
