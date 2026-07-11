from src.market_filter import adjust_stock_position, derive_market_filter
def test_exposure_is_bounded_and_uncertain_supported():
    p={h:{"Bull":.25,"Sideway":.25,"Bear":.25,"Stress":.25} for h in (20,40,60)}
    out=derive_market_filter(p,"Uncertain",.2); assert out["market_state"]=="Uncertain" and 0<=out["exposure_multiplier"]<=1
    assert adjust_stock_position(100,out["exposure_multiplier"],out["market_state"],"Uncertain")<=100

