<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant as Schedule;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;

#[Group('Layers')]
#[CoversClass(Dense::class)]
class DenseTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var positive-int
     */
    protected int $fanIn;

    /**
     * @var Matrix
     */
    protected Matrix $x;

    /**
     * @var Deferred
     */
    protected Deferred $prevGrad;

    /**
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * @var Dense
     */
    protected Dense $layer;

    protected function setUp() : void
    {
        $this->fanIn = 3;

        $this->x = Matrix::quick([
            [1.0, 2.5, -0.1],
            [0.1, 0.0, 3.0],
            [0.002, -6.0, -0.5],
        ]);

        $this->prevGrad = new Deferred(function () {
            return Matrix::quick([
                [0.50, 0.2, 0.01],
                [0.25, 0.1, 0.89],
            ]);
        });

        $this->optimizer = new Stochastic(new Schedule(0.001));

        $this->layer = new Dense(
            2,
            bias: true,
            weightInitializer: new He(),
            biasInitializer: new Constant(0.0)
        );

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(Dense::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    #[Test]
    public function negativeL1Penalty() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new Dense(2, l1Penalty: -1.0);
    }

    #[Test]
    public function l1Penalty() : void
    {
        srand(self::RANDOM_SEED);

        $layer = new Dense(
            2,
            0.5,
            bias: true,
            weightInitializer: new He(),
            biasInitializer: new Constant(0.0)
        );

        $layer->initialize($this->fanIn);

        $this->assertEquals(2, $layer->width());

        $expected = [
            [0.1655431527858090, -3.3067210399720404, 0.4696824341238931],
            [1.0048212865204000, -3.6402121844350117, 0.2683915072737035],
        ];

        $forward = $layer->forward($this->x);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $expected = [
            [0.3124653598013489, 0.1249861439205396, 0.8680008614843256],
            [0.2039667923312287, 0.0815867169324915, 0.2613122897726918],
            [0.5574294942663576, 0.2229717977065430, 0.9071430347096159],
        ];

        $gradient = $layer->back($this->prevGrad)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        foreach ($layer->parameters() as $param) {
            if ($param->hasGradient()) {
                $this->optimizer->warm($param);

                $this->optimizer->update($param);
            }
        }

        $expected = [
            [0.1632775607858090, -3.3154025399720405, 0.4670303341238930],
            [1.0023518755203999, -3.6469966844350115, 0.2573853572737035],
        ];

        $infer = $layer->infer($this->x);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    #[Test]
    public function l1AndL2Penalties() : void
    {
        srand(self::RANDOM_SEED);

        $layer = new Dense(
            2,
            0.5,
            0.5,
            bias: true,
            weightInitializer: new He(),
            biasInitializer: new Constant(0.0)
        );

        $layer->initialize($this->fanIn);

        $this->assertEquals(2, $layer->width());

        $expected = [
            [0.1655431527858090, -3.3067210399720404, 0.4696824341238931],
            [1.0048212865204000, -3.6402121844350117, 0.2683915072737035],
        ];

        $forward = $layer->forward($this->x);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $expected = [
            [0.3124653598013489, 0.1249861439205396, 0.8680008614843256],
            [0.2039667923312287, 0.0815867169324915, 0.2613122897726918],
            [0.5574294942663576, 0.2229717977065430, 0.9071430347096159],
        ];

        $gradient = $layer->back($this->prevGrad)->compute();

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        foreach ($layer->parameters() as $param) {
            if ($param->hasGradient()) {
                $this->optimizer->warm($param);

                $this->optimizer->update($param);
            }
        }

        $expected = [
            [0.1631947892094160, -3.3137491794520552, 0.4667954929068314],
            [1.0018494648771400, -3.6451765783427944, 0.2572511615200666],
        ];

        $infer = $layer->infer($this->x);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }

    #[Test]
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize($this->fanIn);

        $this->assertEquals(2, $this->layer->width());

        $expected = [
            [0.1655431527858090, -3.3067210399720404, 0.4696824341238931],
            [1.0048212865204000, -3.6402121844350117, 0.2683915072737035],
        ];

        $forward = $this->layer->forward($this->x);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEqualsWithDelta($expected, $forward->asArray(), 1e-8);

        $gradient = $this->layer->back($this->prevGrad)->compute();

        $expected = [
            [0.3124653598013489, 0.1249861439205396, 0.8680008614843256],
            [0.2039667923312287, 0.0815867169324915, 0.2613122897726918],
            [0.5574294942663576, 0.2229717977065430, 0.9071430347096159],
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEqualsWithDelta($expected, $gradient->asArray(), 1e-8);

        foreach ($this->layer->parameters() as $param) {
            if ($param->hasGradient()) {
                $this->optimizer->warm($param);

                $this->optimizer->update($param);
            }
        }

        $expected = [
            [0.1638285607858090, -3.3171525399720405, 0.4682303341238930],
            [1.0029028755203999, -3.6487466844350114, 0.2585853572737035],
        ];

        $infer = $this->layer->infer($this->x);

        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEqualsWithDelta($expected, $infer->asArray(), 1e-8);
    }
}
