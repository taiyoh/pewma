package pewma

import (
	"errors"
	"math"
)

// Value represents acceptable and calculatable.
type Value float64

func (v Value) sqrt() float64 {
	return math.Sqrt(float64(v))
}

func (v Value) square() Value {
	return v * v
}

// Config represents coefficients for calculating specified time series.
type Config struct {
	trainingPeriod int     // T
	alpha0Weight   float64 // α
	betaWeight     float64 // β
}

// NewConfig returns Config object from supplied values with validation.
func NewConfig(t int, alpha0 float64, beta float64) (Config, error) {
	if alpha0 <= 0 || 1 <= alpha0 {
		return Config{}, errors.New("alpha0 weight is invalid")
	}
	if beta < 0 || 1 <= beta {
		return Config{}, errors.New("beta weight is invalid")
	}
	return Config{
		trainingPeriod: t,
		alpha0Weight:   adapt,
		betaWeight:     beta,
	}, nil
}

type factors struct {
	s1           Value
	s2           Value
	stdDeviation float64
}

func (f factors) zt(v Value) float64 {
	// Xˆt+1 ← s1
	// Zt ← Xt−Xˆt
	return float64(v-f.s1) / f.stdDeviation
}

func (f factors) pt(v Value) float64 {
	// Pt ← exp(-(Zt^2)/2)/√2π
	zt := f.zt(v)
	return math.Exp(-(zt*zt)/2) / math.Sqrt(2*math.Pi)
}

func (f factors) detectAnomaly(threshold float64, v Value) bool {
	return f.pt(v) <= threshold
}

func (f factors) New(alpha Value, v Value) factors {
	/*
			s1 ← αts1 + (1 − αt)Xt
			s2 ← αts2 + (1 − αt)Xt^2
		    σˆt+1 ←√s2 − s1^2
	*/
	s1 := alpha*f.s1 + (1-alpha)*v
	s2 := alpha*f.s2 + (1-alpha)*v.square()
	stdDeviation := (s2 - s1.square()).sqrt()
	return factors{
		s1:           s1,
		s2:           s2,
		stdDeviation: stdDeviation,
	}
}

// PEWMA is Probabilistic EWMA (Exponentially Weighted Moving Average) algorithm.
type PEWMA struct {
	captured []Value
	config   Config
	factors  factors
}

// New returns PEWMA object.
func New(config Config) *PEWMA {
	return &PEWMA{
		captured: []Value{},
		config:   config,
		factors:  factors{},
	}
}

func (p *PEWMA) alpha(v Value) Value {
	/*
	   if t ≤ T then
	       αt ← 1 − 1/t
	   else
	       αt ← (1 − βPt)α
	   end if
	*/
	conf := p.config
	if l := len(p.captured); l < conf.trainingPeriod {
		return Value(1 - 1.0/(l+1))
	}
	f := p.factors
	return Value((1 - conf.betaWeight*f.pt(v)) * conf.alpha0Weight)
}

// Status represents analyzing status from supplied value.
type Status int

const (
	// InTraining represents captured data count has not reached training period.
	InTraining Status = iota
	// InOrdinary represents supplied data is in range of standard deviation.
	InOrdinary
	// Outlier represents supplied data is out of standard deviation.
	Outlier
)

// Analyze caluclates Status from supplied value and threshold.
func (p *PEWMA) Analyze(v Value, threshold float64) Status {
	conf := p.config
	if len(p.captured) < conf.trainingPeriod {
		return InTraining
	}
	if p.factors.detectAnomaly(threshold, v) {
		return Outlier
	}
	return InOrdinary
}

func (p *PEWMA) pushAndPop(v Value) []Value {
	newCaptured := append(p.captured, v)
	if len(newCaptured) > p.config.trainingPeriod {
		return newCaptured[1:]
	}
	return newCaptured
}

// Add provides capture supplied value and new PEWMA object.
func (p *PEWMA) Add(v Value) *PEWMA {
	alpha := p.alpha(v)

	return &PEWMA{
		config:   p.config,
		captured: p.pushAndPop(v),
		factors:  p.factors.New(alpha, v),
	}
}
